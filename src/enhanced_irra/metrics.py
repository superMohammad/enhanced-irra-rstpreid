"""Identity-relevant text→image ranking, without camera exclusions or image duplication."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RetrievalMetrics:
    rank1: float
    rank5: float
    rank10: float
    mAP: float
    mINP: float
    queries: int
    gallery: int


def retrieval_metrics(similarity: Tensor, query_ids: Tensor, gallery_ids: Tensor) -> RetrievalMetrics:
    """Full-gallery AP/INP; percentages; stable gallery-order ties; every query must have positives.

    AP is mean precision at every relevant rank. INP is number of relevant gallery
    images divided by the rank of the last relevant image. No cross-camera exclusion
    is applied in the official IRRA text-to-image protocol.
    """
    if similarity.ndim != 2 or similarity.shape != (query_ids.numel(), gallery_ids.numel()):
        raise ValueError("Similarity must have shape number_of_queries × number_of_gallery_images")
    if similarity.numel() == 0 or not bool(torch.isfinite(similarity).all()):
        raise ValueError("Cannot evaluate empty or nonfinite similarities")
    order: Tensor = similarity.detach().cpu().argsort(dim=1, descending=True, stable=True)
    matches: Tensor = gallery_ids.cpu()[order].eq(query_ids.cpu()[:, None])
    relevant: Tensor = matches.sum(1)
    if bool(relevant.eq(0).any()):
        raise ValueError("Every query must have a relevant gallery identity; inspect split/filtering")
    positions: Tensor = torch.arange(1, matches.shape[1] + 1, dtype=torch.float64)
    precision: Tensor = matches.cumsum(1) / positions
    ap: Tensor = (precision * matches).sum(1) / relevant
    last: Tensor = (matches * positions).max(dim=1).values
    cmc: list[float] = [100 * float(matches[:, :min(k, matches.shape[1])].any(1).float().mean()) for k in (1, 5, 10)]
    return RetrievalMetrics(cmc[0], cmc[1], cmc[2], 100 * float(ap.mean()), 100 * float((relevant / last).mean()), matches.shape[0], matches.shape[1])
