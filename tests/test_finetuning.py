"""Warm starts must preserve trained weights, identity labels and model semantics."""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pytest
from torch import nn

from enhanced_irra.config import ExperimentConfig, load_config, save_config
from enhanced_irra.data import sha256_file
from enhanced_irra.model import RetrievalModel
from enhanced_irra.training import model_from_run


def prepare_source(root: Path) -> tuple[ExperimentConfig, Path, Path]:
    config: ExperimentConfig = load_config(Path("configs/PRIORITY/E6_seed17.json"))
    checkpoint: Path = root / "best.pt"
    checkpoint.write_bytes(b"verified fixture")
    save_config(config, root / "config.json")
    (root / "summary.json").write_text(json.dumps({"checkpoint_sha256": sha256_file(checkpoint), "best_epoch": 9}))
    manifest: Path = root / "training_manifest.json"
    manifest.write_text(json.dumps({"images": ["first.jpg"], "label_map": {"1": 0}}))
    return config, checkpoint, manifest


def test_warm_start_restores_weights_with_new_optimization_settings(tmp_path: Path) -> None:
    source, checkpoint, manifest = prepare_source(tmp_path)
    config: ExperimentConfig = replace(source, epochs=30, warmup_epochs=2, learning_rate=5e-6, output="new-run")
    model: RetrievalModel = cast(RetrievalModel, Mock(spec=RetrievalModel))
    model.classifier = nn.Linear(512, 3701)
    with patch("enhanced_irra.training.load_checkpoint", return_value=model) as restore:
        restored, epoch, digest = model_from_run(config, checkpoint, 3701, manifest)
    assert restored is model
    assert epoch == 9
    assert digest == sha256_file(checkpoint)
    assert restore.call_args.args[:2] == (checkpoint, source)


def test_warm_start_rejects_objective_change(tmp_path: Path) -> None:
    config, checkpoint, manifest = prepare_source(tmp_path)
    with pytest.raises(ValueError, match="masking"):
        model_from_run(replace(config, masking="random"), checkpoint, 3701, manifest)


def test_warm_start_rejects_changed_identity_labels(tmp_path: Path) -> None:
    config, checkpoint, _ = prepare_source(tmp_path)
    manifest: Path = tmp_path / "new_manifest.json"
    manifest.write_text(json.dumps({"images": ["first.jpg"], "label_map": {"1": 1}}))
    with pytest.raises(ValueError, match="label ordering"):
        model_from_run(config, checkpoint, 3701, manifest)


def test_warm_start_rejects_modified_checkpoint(tmp_path: Path) -> None:
    config, checkpoint, manifest = prepare_source(tmp_path)
    checkpoint.write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="hash mismatch"):
        model_from_run(config, checkpoint, 3701, manifest)
