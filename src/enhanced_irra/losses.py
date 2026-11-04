"""IRRA losses in FP32, including the original predicted-to-target KL direction."""

import torch.nn.functional as F
from torch import Tensor


def identity_distribution(identities: Tensor) -> Tensor:
    """Return B×B targets with unit row sums and all same-ID positives."""
    if identities.ndim != 1 or identities.numel() == 0:
        raise ValueError("Identities must be a nonempty one-dimensional tensor")
    positives: Tensor = identities[:, None].eq(identities[None, :]).float()
    return positives / positives.sum(dim=1, keepdim=True)


def sdm_loss(images: Tensor, texts: Tensor, identities: Tensor, temperature: float) -> Tensor:
    r"""Sum both directions of KL(p || q+1e-8), matching official compute_sdm.

    Inputs: image/text B×512, identities B. p=softmax(cosine/temperature).
    The stabilized target is not renormalized after adding epsilon, as in IRRA.
    """
    if images.shape != texts.shape or images.ndim != 2 or images.shape[0] != identities.numel():
        raise ValueError("SDM requires equal B×D features and B identities")
    if temperature <= 0:
        raise ValueError("SDM temperature must be positive")
    target: Tensor = identity_distribution(identities)
    logits: Tensor = F.normalize(images.float(), dim=-1) @ F.normalize(texts.float(), dim=-1).T / temperature
    directions: tuple[Tensor, Tensor] = (logits, logits.T)
    terms: list[Tensor] = [
        (F.softmax(scores, dim=1) * (F.log_softmax(scores, dim=1) - (target + 1e-8).log())).sum(dim=1).mean()
        for scores in directions
    ]
    return terms[0] + terms[1]


def id_loss(image_logits: Tensor, text_logits: Tensor, identities: Tensor) -> Tensor:
    """IRRA uses one shared classifier and averages the two cross entropies."""
    return (F.cross_entropy(image_logits.float(), identities) + F.cross_entropy(text_logits.float(), identities)) / 2


def mlm_loss(logits: Tensor, targets: Tensor, token_weights: Tensor) -> Tensor:
    """Weighted CE on selected original tokens only; -100 marks ignored positions.

    Empty selections deliberately contribute differentiable zero. Attribute/phrase
    policies never substitute random or stopword candidates when none are present.
    """
    selected: Tensor = targets.ne(-100)
    if logits.shape[:-1] != targets.shape or token_weights.shape != (logits.shape[-1],):
        raise ValueError("MLM expects B×L×V logits, B×L targets, and V token weights")
    if not bool(selected.any()):
        return logits.sum() * 0
    labels: Tensor = targets[selected]
    weights: Tensor = token_weights[labels].float()
    errors: Tensor = F.cross_entropy(logits[selected].float(), labels, reduction="none")
    return (errors * weights).sum() / weights.sum()


def mlm_accuracy(logits: Tensor, targets: Tensor) -> tuple[int, int]:
    """Return correct and selected counts for token-weighted epoch aggregation."""
    selected: Tensor = targets.ne(-100)
    return int((logits.argmax(-1)[selected] == targets[selected]).sum()), int(selected.sum())
