"""Strict RSTPReid parsing and image I/O; official identity-disjoint splits only."""

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class ImageRecord:
    identity: int
    path: Path
    captions: tuple[str, str]
    split: Split
    camera: int


@dataclass(frozen=True)
class DatasetAudit:
    root: str
    annotation_sha256: str
    image_manifest_sha256: str
    identities: dict[str, int]
    images: dict[str, int]
    captions: dict[str, int]
    same_split_duplicate_groups: int
    cross_split_duplicate_groups: int
    verified_images: int


def sha256_file(path: Path) -> str:
    """Stream file hashes without retaining dataset contents in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_dataset(root: Path) -> Path:
    """Find exactly one annotation directory; ambiguity and missing data are errors."""
    matches: list[Path] = sorted(root.rglob("data_captions.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one data_captions.json under {root}, found {matches}. Extract the official archive into data/RSTPReid.")
    location: Path = matches[0].parent
    if not (location / "imgs").is_dir():
        raise FileNotFoundError(f"Missing image directory: {location / 'imgs'}")
    return location


def read_records(root: Path) -> tuple[ImageRecord, ...]:
    """Validate annotation schema, safe paths, camera IDs and all referenced files."""
    location: Path = inspect_dataset(root)
    raw: object = json.loads((location / "data_captions.json").read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("RSTPReid annotations must be a nonempty JSON list")
    records: list[ImageRecord] = []
    image_root: Path = (location / "imgs").resolve()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not {"id", "img_path", "captions", "split"}.issubset(item):
            raise ValueError(f"Invalid annotation schema at row {index}")
        identity: object = item["id"]
        name: object = item["img_path"]
        captions: object = item["captions"]
        split: object = item["split"]
        if type(identity) is not int or identity < 0 or not isinstance(name, str):
            raise ValueError(f"Invalid ID/path at annotation row {index}")
        if not isinstance(captions, list) or len(captions) != 2 or any(not isinstance(c, str) or not c.strip() for c in captions):
            raise ValueError(f"Expected exactly two nonempty English captions at row {index}")
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r} at row {index}; refusing to infer a split")
        path: Path = (image_root / name).resolve()
        if not path.is_relative_to(image_root) or not path.is_file():
            raise FileNotFoundError(f"Missing/unsafe image at row {index}: {path}")
        camera_match: re.Match[str] | None = re.search(r"_c(\d+)_", path.name)
        if camera_match is None or not 1 <= int(camera_match[1]) <= 15:
            raise ValueError(f"Cannot extract official camera 1–15 from {path.name}")
        records.append(ImageRecord(identity, path, (captions[0], captions[1]), cast(Split, split), int(camera_match[1])))
    return tuple(records)


def audit_dataset(root: Path, records: Sequence[ImageRecord]) -> DatasetAudit:
    """Audit all files, decoded pixels, identity/path overlap and official split counts.

    Hash decoded RGB pixels as well as file bytes to catch re-encoded duplicates.
    Same-split duplicates are reported; cross-split duplicates always raise.
    """
    expected: dict[str, int] = {"train": 3701, "val": 200, "test": 200}
    identity_splits: dict[int, set[str]] = {}
    path_splits: dict[Path, set[str]] = {}
    digest_splits: dict[str, list[str]] = {}
    manifest: list[str] = []
    for record in records:
        identity_splits.setdefault(record.identity, set()).add(record.split)
        path_splits.setdefault(record.path, set()).add(record.split)
        with Image.open(record.path) as source:
            image: Image.Image = source.convert("RGB")
            digest: str = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        digest_splits.setdefault(digest, []).append(record.split)
        manifest.append(f"{record.path.name}:{sha256_file(record.path)}:{record.split}:{record.identity}")
    if any(len(splits) > 1 for splits in identity_splits.values()):
        raise ValueError("Identity leakage across official splits")
    if any(len(splits) > 1 for splits in path_splits.values()) or len(path_splits) != len(records):
        raise ValueError("Duplicate image path or image path leakage in annotations")
    cross_duplicates: int = sum(len(set(splits)) > 1 for splits in digest_splits.values())
    counts: dict[str, int] = {split: len({r.identity for r in records if r.split == split}) for split in expected}
    if counts != expected:
        raise ValueError(f"Official identity counts mismatch: expected={expected}, actual={counts}")
    images_per_id: Counter[int] = Counter(r.identity for r in records)
    if set(images_per_id.values()) != {5}:
        raise ValueError("Official RSTPReid requires exactly five images per identity")
    images: dict[str, int] = {split: sum(r.split == split for r in records) for split in expected}
    location: Path = inspect_dataset(root)
    return DatasetAudit(str(location), sha256_file(location / "data_captions.json"), hashlib.sha256("\n".join(sorted(manifest)).encode()).hexdigest(), counts, images, {k: 2 * v for k, v in images.items()}, sum(len(s) > 1 and len(set(s)) == 1 for s in digest_splits.values()), cross_duplicates, len(records))


def training_labels(records: Sequence[ImageRecord]) -> dict[int, int]:
    """Map sorted training IDs only; validation/test IDs never enter classifiers."""
    return {identity: index for index, identity in enumerate(sorted({r.identity for r in records if r.split == "train"}))}


def training_transform(height: int, width: int) -> Callable[[Image.Image], Tensor]:
    """Match IRRA --img_aug preprocessing; train-only stochastic transforms."""
    return cast(Callable[[Image.Image], Tensor], T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(0.5), T.Pad(10), T.RandomCrop((height, width)),
        T.ToTensor(), T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        T.RandomErasing(p=0.5, scale=(0.02, 0.4), value=(0.48145466, 0.4578275, 0.40821073)),
    ]))


def evaluation_transform(height: int, width: int) -> Callable[[Image.Image], Tensor]:
    """Deterministic rectangular CLIP normalization matching IRRA evaluation."""
    return cast(Callable[[Image.Image], Tensor], T.Compose([
        T.Resize((height, width), interpolation=InterpolationMode.BILINEAR), T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]))


class ImageConnector(Dataset[tuple[Tensor, int]]):
    """PyTorch Dataset interface to on-disk gallery images, once per image."""

    def __init__(self, records: tuple[ImageRecord, ...], transform: Callable[[Image.Image], Tensor]) -> None:
        self.records: tuple[ImageRecord, ...] = records
        self.transform: Callable[[Image.Image], Tensor] = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record: ImageRecord = self.records[index]
        with Image.open(record.path) as image:
            return self.transform(image.convert("RGB")), record.identity


class TrainingConnector(Dataset[tuple[Tensor, int, int]]):
    """Image filesystem connector; each index identifies one caption/image pair."""

    def __init__(self, records: tuple[ImageRecord, ...], labels: dict[int, int], transform: Callable[[Image.Image], Tensor]) -> None:
        if not records or any(r.split != "train" for r in records):
            raise ValueError("TrainingConnector accepts nonempty training records only")
        self.records: tuple[ImageRecord, ...] = records
        self.labels: dict[int, int] = dict(labels)
        self.transform: Callable[[Image.Image], Tensor] = transform

    def __len__(self) -> int:
        return 2 * len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int, int]:
        record: ImageRecord = self.records[index // 2]
        with Image.open(record.path) as image:
            tensor: Tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[record.identity], index


def seed_worker(worker_id: int) -> None:
    """Explicit DataLoader callback: seed NumPy/Python from PyTorch worker seed."""
    worker_seed: int = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def verify_training_isolation(train: Sequence[ImageRecord], evaluation: Sequence[ImageRecord]) -> None:
    """Refuse optimization if any training RGB image occurs in validation or test."""
    def digest(record: ImageRecord) -> str:
        with Image.open(record.path) as image:
            rgb: Image.Image = image.convert("RGB")
            return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()

    evaluation_hashes: set[str] = {digest(record) for record in evaluation}
    leaks: list[str] = [str(record.path) for record in train if digest(record) in evaluation_hashes]
    if leaks:
        raise ValueError(f"Training blocked by cross-split image leakage: {leaks}. Resolve quarantine policy explicitly; official split labels must not be changed.")
