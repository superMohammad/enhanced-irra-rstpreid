"""Explicit, immutable experiment settings; no test split is available to training."""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal, cast

Masking = Literal["random", "attribute", "phrase"]
Mode = Literal["SMOKE_TEST", "EXPERIMENT", "FULL_TRAIN"]
RetrievalInput = Literal["clean", "corrupted"]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    mode: Mode
    seed: int
    device: str
    precision: Literal["float32", "float16", "bfloat16"]
    dataset: str
    quarantine_manifest: str
    pretrained: str
    output: str
    epochs: int
    batch_size: int
    accumulation: int
    workers: int
    learning_rate: float
    head_lr_factor: float
    weight_decay: float
    warmup_epochs: int
    patience: int
    gradient_clip: float
    temperature: float
    sdm_weight: float
    id_weight: float
    mlm_weight: float
    masking: Masking
    mask_probability: float
    frequency_weighted: bool
    frequency_power: float
    retrieval_input: RetrievalInput
    train_id_limit: int
    max_steps: int
    cross_depth: int
    image_height: int
    image_width: int

    def __post_init__(self) -> None:
        if self.mode not in ("SMOKE_TEST", "EXPERIMENT", "FULL_TRAIN"):
            raise ValueError(f"Unknown research mode: {self.mode}")
        if self.masking not in ("random", "attribute", "phrase"):
            raise ValueError(f"Unknown masking strategy: {self.masking}")
        if self.retrieval_input not in ("clean", "corrupted"):
            raise ValueError(f"Unknown retrieval input: {self.retrieval_input}")
        if self.precision not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"Unknown training precision: {self.precision}")
        if self.device == "cpu" and self.precision != "float32":
            raise ValueError("CPU research checks require explicitly configured float32")
        if min(self.epochs, self.batch_size, self.accumulation, self.patience, self.cross_depth) < 1:
            raise ValueError("Epochs, batch size, accumulation, patience and cross depth must be positive")
        if min(self.workers, self.train_id_limit, self.max_steps, self.warmup_epochs, self.seed) < 0:
            raise ValueError("Workers, subset limits, steps, warmup and seed cannot be negative")
        if self.temperature <= 0 or not 0 < self.mask_probability < 1:
            raise ValueError("Temperature must be positive and mask_probability must be in (0, 1)")
        if min(self.sdm_weight, self.id_weight, self.mlm_weight, self.weight_decay) < 0:
            raise ValueError("Loss weights and weight decay cannot be negative")
        if min(self.learning_rate, self.head_lr_factor, self.gradient_clip, self.frequency_power) <= 0:
            raise ValueError("Learning rates, clipping and frequency exponent must be positive")
        if self.image_height % 16 or self.image_width % 16:
            raise ValueError("Image dimensions must be multiples of the ViT-B/16 patch size")
        if self.mode == "FULL_TRAIN" and (self.train_id_limit or self.max_steps):
            raise ValueError("FULL_TRAIN cannot truncate the training identities or steps")
        if self.mode != "SMOKE_TEST" and self.device != "cuda":
            raise ValueError("Research training requires explicit CUDA; use SMOKE_TEST for CPU verification")


def load_config(path: Path) -> ExperimentConfig:
    """Reject missing/unknown settings and incorrectly typed JSON values."""
    raw: object = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {path}")
    expected: set[str] = {field.name for field in fields(ExperimentConfig)}
    if set(raw) != expected:
        raise ValueError(f"Config keys mismatch in {path}: missing={expected - set(raw)}, extra={set(raw) - expected}")
    integer_keys: set[str] = {"seed", "epochs", "batch_size", "accumulation", "workers", "warmup_epochs", "patience", "train_id_limit", "max_steps", "cross_depth", "image_height", "image_width"}
    float_keys: set[str] = {"learning_rate", "head_lr_factor", "weight_decay", "gradient_clip", "temperature", "sdm_weight", "id_weight", "mlm_weight", "mask_probability", "frequency_power"}
    for key, value in raw.items():
        required: type = int if key in integer_keys else float if key in float_keys else bool if key == "frequency_weighted" else str
        if type(value) is not required:
            raise TypeError(f"Config {path}, key {key}: expected {required.__name__}, got {type(value).__name__}")
    return ExperimentConfig(**cast(dict[str, str | int | float | bool], raw))  # type: ignore[arg-type]


def save_config(config: ExperimentConfig, path: Path) -> None:
    """Store the exact resolved settings with each independent run."""
    path.write_text(json.dumps(asdict(config), indent=2) + "\n")
