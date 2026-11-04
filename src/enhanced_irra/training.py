"""Explicit training side effects: seeded PyTorch, optimizer, checkpoints and validation."""

import csv
import json
import math
import os
import platform
import random
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from enhanced_irra.config import ExperimentConfig, load_config, save_config
from enhanced_irra.data import (
    ImageRecord,
    TrainingConnector,
    audit_dataset,
    inspect_dataset,
    read_records,
    seed_worker,
    sha256_file,
    training_labels,
    training_transform,
    verify_training_isolation,
)
from enhanced_irra.evaluation import evaluate, save_evaluation, synchronize
from enhanced_irra.losses import id_loss, mlm_accuracy, mlm_loss, sdm_loss
from enhanced_irra.masking import (
    EncodedCaption,
    MaskedCaption,
    TokenizerConnector,
    frequency_weights,
    mask_caption,
    tokenizer_connector,
)
from enhanced_irra.model import GlobalFeatures, RetrievalModel, load_pretrained


class EpochRow(TypedDict):
    epoch: int
    total_loss: float
    sdm_loss: float
    id_loss: float
    mlm_loss: float
    mlm_accuracy: float | None
    selected_tokens: int
    effective_mask_rate: float
    visual_mask_fraction: float
    stopword_mask_fraction: float
    phrase_coverage: float
    validation_rank1: float
    validation_mAP: float
    epoch_seconds: float
    peak_gpu_memory_bytes: int
    optimizer_updates: int


def seed_runtime(seed: int) -> None:
    """Configure external random/CUDA state explicitly; unsupported deterministic ops raise."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def training_records(records: tuple[ImageRecord, ...], config: ExperimentConfig) -> tuple[ImageRecord, ...]:
    """Apply an explicitly authorized quarantine manifest; never modify original annotations."""
    excluded: set[str] = set()
    if config.quarantine_manifest:
        manifest: object = json.loads(Path(config.quarantine_manifest).read_text())
        if not isinstance(manifest, dict) or manifest.get("policy") != "exclude_training_duplicates" or manifest.get("authorized") is not True:
            raise ValueError("Quarantine manifest must record explicit authorization and exclude_training_duplicates policy")
        annotation: Path = inspect_dataset(Path(config.dataset)) / "data_captions.json"
        if manifest.get("annotation_sha256") != sha256_file(annotation):
            raise ValueError("Quarantine manifest targets a different annotation version")
        paths: object = manifest.get("training_paths")
        if not isinstance(paths, list) or any(not isinstance(p, str) for p in paths):
            raise TypeError("Quarantine training_paths must be an explicit list of training filenames")
        excluded = set(paths)
        original_train: set[str] = {r.path.name for r in records if r.split == "train"}
        if not excluded.issubset(original_train):
            raise ValueError("Quarantine must reference only existing training images")
    train: tuple[ImageRecord, ...] = tuple(r for r in records if r.split == "train" and r.path.name not in excluded)
    if config.train_id_limit:
        ids: set[int] = set(sorted({r.identity for r in train})[:config.train_id_limit])
        train = tuple(r for r in train if r.identity in ids)
    verify_training_isolation(train, tuple(r for r in records if r.split != "train"))
    return train


def optimizer_for(model: RetrievalModel, config: ExperimentConfig) -> AdamW:
    """AdamW with IRRA-style higher LR for random heads; bias/LN have no decay."""
    parameters: list[dict[str, object]] = []
    for name, parameter in model.named_parameters():
        is_backbone: bool = name.startswith("backbone.")
        factor: float = 1.0 if is_backbone else config.head_lr_factor
        parameters.append({"params": [parameter], "lr": config.learning_rate * factor, "weight_decay": config.weight_decay if parameter.ndim > 1 else 0.0})
    return AdamW(parameters, lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8)


def schedule_factor(epoch: int, warmup: int, total: int) -> float:
    if warmup and epoch < warmup:
        return 0.1 + 0.9 * epoch / warmup
    progress: float = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def create_model(config: ExperimentConfig, classes: int) -> RetrievalModel:
    return RetrievalModel(load_pretrained(Path(config.pretrained), (config.image_height, config.image_width)), classes, config.cross_depth)


def save_checkpoint(model: RetrievalModel, config: ExperimentConfig, classes: int, epoch: int, path: Path) -> str:
    """Write complete inference weights atomically; failed writes never replace best.pt."""
    temporary: Path = path.with_suffix(".tmp")
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "classes": classes, "epoch": epoch, "config_json": json.dumps(asdict(config))}, temporary)
    temporary.replace(path)
    return sha256_file(path)


def load_checkpoint(path: Path, config: ExperimentConfig, device: torch.device) -> RetrievalModel:
    """Strict restore; config must match the checkpoint, including data and experiment."""
    payload: dict[str, object] = torch.load(path, map_location="cpu", weights_only=True)
    if payload["config_json"] != json.dumps(asdict(config)):
        raise ValueError(f"Checkpoint/config mismatch: {path}")
    classes: object = payload["classes"]
    if type(classes) is not int:
        raise TypeError(f"Invalid class count in checkpoint: {path}")
    model: RetrievalModel = create_model(config, classes)
    model.load_state_dict(cast(dict[str, Tensor], payload["model"]), strict=True)
    return model.to(device).eval()


def write_history(rows: Sequence[EpochRow], directory: Path) -> None:
    (directory / "history.json").write_text(json.dumps(list(rows), indent=2) + "\n")
    with (directory / "history.csv").open("w", newline="") as stream:
        writer: csv.DictWriter[str] = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_from_run(config: ExperimentConfig, checkpoint: Path, classes: int, manifest: Path) -> tuple[RetrievalModel, int, str]:
    """Restore verified weights for a new optimization stage with unchanged E6 semantics.

    Optimization settings may change, but architecture, objectives, population and
    identity label ordering must match. This is a warm start, not optimizer resume.
    """
    source: ExperimentConfig = load_config(checkpoint.parent / "config.json")
    compatible_fields: tuple[str, ...] = ("name", "dataset", "quarantine_manifest", "pretrained", "cross_depth", "image_height", "image_width", "masking", "frequency_weighted", "frequency_power", "mask_probability", "retrieval_input", "train_id_limit", "temperature", "sdm_weight", "id_weight", "mlm_weight")
    for field in compatible_fields:
        if getattr(source, field) != getattr(config, field):
            raise ValueError(f"Warm-start configuration mismatch: field={field}, checkpoint={checkpoint}")
    if json.loads((checkpoint.parent / "training_manifest.json").read_text()) != json.loads(manifest.read_text()):
        raise ValueError(f"Warm-start training population or label ordering mismatch: {checkpoint}")
    summary: dict[str, object] = json.loads((checkpoint.parent / "summary.json").read_text())
    digest: str = sha256_file(checkpoint)
    if digest != summary["checkpoint_sha256"]:
        raise ValueError(f"Warm-start checkpoint hash mismatch: checkpoint={checkpoint}, actual={digest}")
    epoch: object = summary["best_epoch"]
    if type(epoch) is not int:
        raise TypeError(f"Invalid source checkpoint epoch: {checkpoint}")
    model: RetrievalModel = load_checkpoint(checkpoint, source, torch.device("cpu"))
    if model.classifier.out_features != classes:
        raise ValueError(f"Warm-start classifier class count mismatch: {checkpoint}")
    return model, epoch, digest


def train(config: ExperimentConfig, initialization: Path | None) -> Path:
    """Run one experiment, select only on validation, and verify the saved checkpoint.

    Accumulation changes optimizer batch size, but SDM negatives are scoped to each
    microbatch. Never describe accumulated microbatches as an equivalent SDM batch.
    An explicit initialization starts a new fine-tuning stage with fresh optimizer
    state. The initial checkpoint participates in validation selection as epoch 0.
    """
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but torch.cuda.is_available() is False. Repair the NVIDIA driver/device access; no CPU fallback is applied.")
    device: torch.device = torch.device(config.device)
    if config.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured bfloat16 training is unsupported by this CUDA device")
    directory: Path = Path(config.output) / config.name / f"seed{config.seed}"
    if directory.exists():
        raise FileExistsError(f"Run already exists: {directory}. Choose a new output directory to preserve evidence.")
    records: tuple[ImageRecord, ...] = read_records(Path(config.dataset))
    audit = audit_dataset(Path(config.dataset), records)
    train_records: tuple[ImageRecord, ...] = training_records(records, config)
    val_records: tuple[ImageRecord, ...] = tuple(r for r in records if r.split == "val")
    if config.mode == "SMOKE_TEST":
        val_ids: set[int] = set(sorted({r.identity for r in val_records})[:4])
        val_records = tuple(r for r in val_records if r.identity in val_ids)
    if len(train_records) < 2:
        raise ValueError("Training needs at least two images; inspect identity limits")
    directory.mkdir(parents=True)
    save_config(config, directory / "config.json")
    (directory / "dataset_audit.json").write_text(json.dumps(asdict(audit), indent=2) + "\n")
    source_root: Path = Path(__file__).parent
    (directory / "source_manifest.json").write_text(json.dumps({str(path.relative_to(source_root)): sha256_file(path) for path in sorted(source_root.rglob("*.py"))}, indent=2) + "\n")
    tokenizer: TokenizerConnector = tokenizer_connector()
    captions: tuple[EncodedCaption, ...] = tuple(tokenizer.encode(c) for r in train_records for c in r.captions)
    weights: Tensor = frequency_weights(train_records, tokenizer, config.frequency_power) if config.frequency_weighted else torch.ones(49408)
    torch.save(weights, directory / "token_weights.pt")
    seed_runtime(config.seed)
    labels: dict[int, int] = training_labels(records)
    (directory / "training_manifest.json").write_text(json.dumps({"images": [r.path.name for r in train_records], "identities": sorted({r.identity for r in train_records}), "label_map": labels}, indent=2))
    source_epoch: int = 0
    source_hash: str = ""
    model: RetrievalModel
    if initialization is None:
        model = create_model(config, len(labels))
    else:
        model, source_epoch, source_hash = model_from_run(config, initialization, len(labels), directory / "training_manifest.json")
    model = model.to(device)
    optimizer: AdamW = optimizer_for(model, config)
    scheduler: LambdaLR = LambdaLR(optimizer, lr_lambda=lambda epoch: schedule_factor(epoch, config.warmup_epochs, config.epochs))
    scaler: GradScaler = GradScaler(device.type, enabled=config.precision == "float16", init_scale=128.0)
    autocast_dtype: torch.dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float16
    generator: torch.Generator = torch.Generator().manual_seed(config.seed)
    loader: DataLoader[tuple[Tensor, int, int]] = DataLoader(TrainingConnector(train_records, labels, training_transform(config.image_height, config.image_width)), batch_size=config.batch_size, shuffle=True, num_workers=config.workers, drop_last=True, generator=generator, worker_init_fn=seed_worker)
    if not len(loader):
        raise ValueError("Training microbatch exceeds available caption pairs")
    weights = weights.to(device)
    history: list[EpochRow] = []
    best: float = -1.0
    best_epoch: int = 0
    best_hash: str = ""
    if initialization is not None:
        initial = evaluate(model, val_records, tokenizer, device, config.batch_size, config.workers, (config.image_height, config.image_width))
        best = initial.metrics.rank1
        best_hash = save_checkpoint(model, config, len(labels), 0, directory / "best.pt")
        save_evaluation(initial, val_records, directory / "validation")
        (directory / "initialization.json").write_text(json.dumps({"checkpoint": str(initialization.resolve()), "checkpoint_sha256": source_hash, "source_epoch": source_epoch, "optimizer": "fresh AdamW", "validation": asdict(initial.metrics)}, indent=2) + "\n")
        print(json.dumps({"event": "warm_start", "source_epoch": source_epoch, "validation": asdict(initial.metrics)}), flush=True)
    stale: int = 0
    total_start: float = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start: float = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sums: list[float] = [0.0, 0.0, 0.0, 0.0]
        correct: int = 0
        selected: int = 0
        visual: int = 0
        stopwords: int = 0
        eligible: int = 0
        phrases: int = 0
        selected_phrases: int = 0
        mask_count: int = 0
        updates: int = 0
        steps: int = min(len(loader), config.max_steps) if config.max_steps else len(loader)
        for step, (images, identities, indices) in enumerate(loader):
            if step >= steps:
                break
            masked: list[MaskedCaption] = [mask_caption(captions[int(index)], config.masking, config.mask_probability, config.seed * 1_000_000_007 + (source_epoch + epoch) * len(captions) + int(index)) for index in indices]
            clean: Tensor = torch.stack([m.clean for m in masked]).to(device)
            corrupted: Tensor = torch.stack([m.corrupted for m in masked]).to(device)
            targets: Tensor = torch.stack([m.targets for m in masked]).to(device)
            retrieval_ids: Tensor = clean if config.retrieval_input == "clean" else corrupted
            identities = identities.to(device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=config.precision != "float32"):
                features: GlobalFeatures = model(images.to(device), retrieval_ids)
                sdm: Tensor = sdm_loss(features.images, features.texts, identities, config.temperature)
                identity: Tensor = id_loss(model.classifier(features.images), model.classifier(features.texts), identities) if config.id_weight else sdm * 0
                mlm: Tensor = sdm * 0
                if config.mlm_weight:
                    logits: Tensor = model.predict_masked(features.image_tokens, corrupted)
                    mlm = mlm_loss(logits, targets, weights)
                    batch_correct, batch_selected = mlm_accuracy(logits.detach(), targets)
                    correct += batch_correct
                    selected += batch_selected
                total: Tensor = config.sdm_weight * sdm + config.id_weight * identity + config.mlm_weight * mlm
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"Nonfinite loss: experiment={config.name}, seed={config.seed}, epoch={epoch}, step={step}")
            # Normalize a final partial accumulation window by its actual size.
            window_start: int = (step // config.accumulation) * config.accumulation
            window_size: int = min(config.accumulation, steps - window_start)
            torch.autograd.backward(scaler.scale(total / window_size))
            if (step + 1) % config.accumulation == 0 or step + 1 == steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            loss_sums = [old + float(value.detach()) for old, value in zip(loss_sums, (total, sdm, identity, mlm), strict=True)]
            visual += sum(m.visual_selected for m in masked)
            stopwords += sum(m.stopword_selected for m in masked)
            eligible += sum(m.eligible_tokens for m in masked)
            phrases += sum(m.phrase_count for m in masked)
            selected_phrases += sum(m.selected_phrase_count for m in masked)
            mask_count += sum(len(m.selected) for m in masked)
        scheduler.step()
        synchronize(device)
        training_seconds: float = time.perf_counter() - epoch_start
        validation = evaluate(model, val_records, tokenizer, device, config.batch_size, config.workers, (config.image_height, config.image_width))
        row: EpochRow = {"epoch": epoch + 1, "total_loss": loss_sums[0] / steps, "sdm_loss": loss_sums[1] / steps, "id_loss": loss_sums[2] / steps, "mlm_loss": loss_sums[3] / steps, "mlm_accuracy": correct / selected if selected else None, "selected_tokens": mask_count, "effective_mask_rate": mask_count / eligible if eligible else 0.0, "visual_mask_fraction": visual / mask_count if mask_count else 0.0, "stopword_mask_fraction": stopwords / mask_count if mask_count else 0.0, "phrase_coverage": selected_phrases / phrases if phrases else 0.0, "validation_rank1": validation.metrics.rank1, "validation_mAP": validation.metrics.mAP, "epoch_seconds": training_seconds, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0, "optimizer_updates": updates}
        history.append(row)
        write_history(history, directory)
        print(json.dumps(row), flush=True)
        if validation.metrics.rank1 > best:
            best = validation.metrics.rank1
            best_epoch = epoch + 1
            stale = 0
            best_hash = save_checkpoint(model, config, len(labels), best_epoch, directory / "best.pt")
            save_evaluation(validation, val_records, directory / "validation")
        else:
            stale += 1
        if stale >= config.patience:
            break
    del optimizer, scheduler, model
    reloaded: RetrievalModel = load_checkpoint(directory / "best.pt", config, device)
    restored = evaluate(reloaded, val_records, tokenizer, device, config.batch_size, config.workers, (config.image_height, config.image_width))
    saved: dict[str, Tensor] = torch.load(directory / "validation" / "embeddings.pt", map_location="cpu", weights_only=True)
    reload_equal: bool = torch.equal(saved["query_features"], restored.query_features) and torch.equal(saved["gallery_features"], restored.gallery_features)
    if not reload_equal:
        raise RuntimeError(f"Checkpoint reload changed deterministic embeddings: {directory}")
    summary: dict[str, object] = {"status": "completed", "experiment": config.name, "mode": config.mode, "seed": config.seed, "best_epoch": best_epoch, "validation": asdict(restored.metrics), "run_seconds": time.perf_counter() - total_start, "training_seconds": sum(row["epoch_seconds"] for row in history), "checkpoint_sha256": best_hash, "checkpoint_reload_equal": reload_equal, "deterministic": restored.deterministic, "parameters": sum(p.numel() for p in reloaded.parameters()), "checkpoint_bytes": (directory / "best.pt").stat().st_size, "torch": torch.__version__, "python": platform.python_version(), "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "microbatch_size": config.batch_size, "accumulation": config.accumulation, "protocol": "decontaminated-training" if config.quarantine_manifest else "official", "train_images": len(train_records), "train_identities": len({r.identity for r in train_records}), "annotation_sha256": audit.annotation_sha256, "image_manifest_sha256": audit.image_manifest_sha256}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return directory
