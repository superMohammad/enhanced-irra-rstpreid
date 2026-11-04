"""Measured, deterministic validation with reusable global gallery embeddings."""

import csv
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from enhanced_irra.data import ImageConnector, ImageRecord, evaluation_transform
from enhanced_irra.masking import TokenizerConnector
from enhanced_irra.metrics import RetrievalMetrics, retrieval_metrics
from enhanced_irra.model import RetrievalModel


@dataclass(frozen=True)
class EvaluationOutput:
    metrics: RetrievalMetrics
    query_features: Tensor
    gallery_features: Tensor
    query_ids: Tensor
    gallery_ids: Tensor
    similarity: Tensor
    encoding_seconds: float
    query_latency_ms: float
    ranking_latency_ms: float
    deterministic: bool


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate(model: RetrievalModel, records: tuple[ImageRecord, ...], tokenizer: TokenizerConnector, device: torch.device, batch_size: int, workers: int, image_size: tuple[int, int]) -> EvaluationOutput:
    """Encode each gallery image once, all two-caption queries, and full rankings.

    Validation/test transforms contain no augmentation. Latency includes warmup
    for bulk encoding but uses a warmed single-query measurement (20 repeats).
    """
    if not records or len({r.split for r in records}) != 1 or records[0].split == "train":
        raise ValueError("Evaluation requires one nonempty official validation/test split")
    model.eval()
    synchronize(device)
    start: float = time.perf_counter()
    gallery: list[Tensor] = []
    gids: list[Tensor] = []
    loader: DataLoader[tuple[Tensor, int]] = DataLoader(ImageConnector(records, evaluation_transform(*image_size)), batch_size=batch_size, shuffle=False, num_workers=workers)
    for images, identities in loader:
        gallery.append(F.normalize(model.encode_image(images.to(device)), dim=-1).cpu())
        gids.append(identities)
    captions: list[str] = [caption for record in records for caption in record.captions]
    tokens: Tensor = torch.tensor([tokenizer.encode(c).token_ids for c in captions], dtype=torch.long)
    queries: list[Tensor] = []
    for offset in range(0, len(captions), batch_size):
        queries.append(F.normalize(model.encode_text(tokens[offset:offset + batch_size].to(device)), dim=-1).cpu())
    synchronize(device)
    duration: float = time.perf_counter() - start
    qfeats: Tensor = torch.cat(queries)
    gfeats: Tensor = torch.cat(gallery)
    query_ids: Tensor = torch.tensor([r.identity for r in records for _ in r.captions], dtype=torch.long)
    gallery_ids: Tensor = torch.cat(gids)
    query: Tensor = tokens[:1].to(device)
    first: Tensor = model.encode_text(query)
    second: Tensor = model.encode_text(query)
    deterministic: bool = torch.equal(first, second)
    if not deterministic:
        raise RuntimeError("Fixed-checkpoint text inference is nondeterministic")
    synchronize(device)
    query_start: float = time.perf_counter()
    for _ in range(20):
        model.encode_text(query)
    synchronize(device)
    latency: float = (time.perf_counter() - query_start) * 1000 / 20
    rank_start: float = time.perf_counter()
    similarity: Tensor = qfeats @ gfeats.T
    metrics: RetrievalMetrics = retrieval_metrics(similarity, query_ids, gallery_ids)
    rank_ms: float = (time.perf_counter() - rank_start) * 1000 / len(captions)
    return EvaluationOutput(metrics, qfeats, gfeats, query_ids, gallery_ids, similarity, duration, latency, rank_ms, deterministic)


def save_evaluation(result: EvaluationOutput, records: Sequence[ImageRecord], directory: Path) -> None:
    """Save measured metrics and normalized galleries, never reference-paper values."""
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"metrics": asdict(result.metrics), "encoding_seconds": result.encoding_seconds, "query_latency_ms": result.query_latency_ms, "ranking_latency_ms": result.ranking_latency_ms, "deterministic": result.deterministic, "gallery_paths": [str(r.path) for r in records], "caption_queries": [c for r in records for c in r.captions]}
    (directory / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (directory / "metrics.csv").open("w", newline="") as stream:
        writer: csv.DictWriter[str] = csv.DictWriter(stream, fieldnames=list(asdict(result.metrics)))
        writer.writeheader()
        writer.writerow(asdict(result.metrics))
    torch.save({"query_features": result.query_features, "gallery_features": result.gallery_features, "query_ids": result.query_ids, "gallery_ids": result.gallery_ids, "similarity": result.similarity}, directory / "embeddings.pt")


def cached_topk(embedding_path: Path, query_embedding: Tensor, count: int) -> tuple[Tensor, Tensor]:
    """Rank a normalized query against a previously measured gallery, for notebook use."""
    payload: dict[str, Tensor] = torch.load(embedding_path, map_location="cpu", weights_only=True)
    gallery: Tensor = payload["gallery_features"]
    if query_embedding.shape != (1, gallery.shape[1]) or not 1 <= count <= len(gallery):
        raise ValueError("Expected one 512D query and a valid top-K within the cached gallery")
    scores: Tensor = F.normalize(query_embedding.float().cpu(), dim=-1) @ gallery.T
    order: Tensor = scores.argsort(dim=1, descending=True, stable=True)[:, :count]
    return scores.gather(1, order), order
