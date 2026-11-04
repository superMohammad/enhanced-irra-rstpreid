"""Typed PyTorch interfaces around pinned, MIT-licensed official IRRA primitives."""

import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from enhanced_irra.data import sha256_file
from enhanced_irra.masking import EOT_ID, VOCAB_SIZE
from enhanced_irra.vendor.clip_model import CLIP, LayerNorm, QuickGELU, ResidualAttentionBlock, Transformer

PRETRAINED_SHA256: str = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


def load_pretrained(path: Path, image_size: tuple[int, int]) -> CLIP:
    """Load official OpenAI ViT-B/16 weights strictly; resize only visual positions.

    Bilinear interpolation matches IRRA. At 384×128, patch sequence is B×193×512;
    text output is B×77×512. No missing weights may remain randomly initialized.
    The checkpoint hash must equal the published OpenAI artifact hash.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing pretrained CLIP weights: {path}. See README download command.")
    digest: str = sha256_file(path)
    if digest != PRETRAINED_SHA256:
        raise ValueError(f"CLIP SHA256 mismatch for {path}: expected={PRETRAINED_SHA256}, actual={digest}")
    pretrained: torch.jit.ScriptModule = torch.jit.load(str(path), map_location="cpu")  # type: ignore[no-untyped-call]
    state: dict[str, Tensor] = {name: tensor.float() for name, tensor in pretrained.state_dict().items() if name not in {"input_resolution", "context_length", "vocab_size", "logit_scale"}}
    backbone: CLIP = CLIP(embed_dim=512, image_resolution=image_size, vision_layers=12, vision_width=768, vision_patch_size=16, stride_size=16, context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12)
    positions: Tensor = state["visual.positional_embedding"]
    height, width = image_size
    if (height, width) != (224, 224):
        grid: Tensor = positions[1:].reshape(1, 14, 14, 768).permute(0, 3, 1, 2)
        resized: Tensor = F.interpolate(grid, size=(height // 16, width // 16), mode="bilinear", align_corners=False)
        state = {**state, "visual.positional_embedding": torch.cat([positions[:1], resized.permute(0, 2, 3, 1).reshape(-1, 768)])}
    backbone.load_state_dict(state, strict=True)
    return backbone.float()


@dataclass(frozen=True)
class GlobalFeatures:
    images: Tensor
    texts: Tensor
    image_tokens: Tensor


class RetrievalModel(nn.Module):
    """PyTorch interface: pretrained global embeddings, shared ID, and IRR MLM.

    Upstream cross-attention has no extra Q residual. Residuals are inside the
    four CLIP-style pre-LN refinement blocks. We retain that exact distinction.
    The classifier is shared across modalities, matching the official code.
    """

    def __init__(self, backbone: CLIP, classes: int, cross_depth: int) -> None:
        super().__init__()
        self.backbone: CLIP = backbone
        self.classifier: nn.Linear = nn.Linear(512, classes)
        self.cross_attn: nn.MultiheadAttention = nn.MultiheadAttention(512, 8, batch_first=True)
        self.cross_modal_transformer: Transformer = Transformer(width=512, layers=cross_depth, heads=8)
        self.ln_pre_t: nn.Module = LayerNorm(512)
        self.ln_pre_i: nn.Module = LayerNorm(512)
        self.ln_post: nn.Module = LayerNorm(512)
        self.mlm_head: nn.Sequential = nn.Sequential(OrderedDict([
            ("dense", nn.Linear(512, 512)), ("gelu", QuickGELU()),
            ("ln", LayerNorm(512)), ("fc", nn.Linear(512, VOCAB_SIZE)),
        ]))
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.zeros_(self.classifier.bias)
        scale: float = 512**-0.5
        projection_std: float = scale * (2 * cross_depth)**-0.5
        fc_std: float = (2 * 512)**-0.5
        for module in self.cross_modal_transformer.resblocks:
            block: ResidualAttentionBlock = cast(ResidualAttentionBlock, module)
            nn.init.normal_(block.attn.in_proj_weight, std=scale)
            nn.init.normal_(block.attn.out_proj.weight, std=projection_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=projection_std)
        nn.init.normal_(self.cross_attn.in_proj_weight, std=scale)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=projection_std)
        nn.init.normal_(cast(nn.Linear, self.mlm_head[0]).weight, std=fc_std)
        nn.init.normal_(cast(nn.Linear, self.mlm_head[3]).weight, std=projection_std)

    def image_tokens(self, images: Tensor) -> Tensor:
        return self.backbone.encode_image(images)

    def text_tokens(self, ids: Tensor) -> Tensor:
        return self.backbone.encode_text(ids)

    def encode_image(self, images: Tensor) -> Tensor:
        return self.image_tokens(images)[:, 0].float()

    def encode_text(self, ids: Tensor) -> Tensor:
        if ids.ndim != 2 or not bool(ids.eq(EOT_ID).sum(1).eq(1).all()):
            raise ValueError("Every caption must contain exactly one CLIP EOT token")
        tokens: Tensor = self.text_tokens(ids)
        return tokens[torch.arange(ids.shape[0], device=ids.device), ids.eq(EOT_ID).long().argmax(1)].float()

    def forward(self, images: Tensor, clean_ids: Tensor) -> GlobalFeatures:
        image_tokens: Tensor = self.image_tokens(images)
        return GlobalFeatures(image_tokens[:, 0].float(), self.encode_text(clean_ids), image_tokens)

    def predict_masked(self, image_tokens: Tensor, corrupted_ids: Tensor) -> Tensor:
        """B×77×512 Q; B×193×512 K/V → B×77×49408 vocabulary logits."""
        queries: Tensor = self.ln_pre_t(self.text_tokens(corrupted_ids))
        keys: Tensor = self.ln_pre_i(image_tokens)
        attended: Tensor = self.cross_attn(queries, keys, keys, need_weights=False)[0]
        refined: Tensor = self.cross_modal_transformer(attended.permute(1, 0, 2)).permute(1, 0, 2)
        return cast(Tensor, self.mlm_head(self.ln_post(refined)))

    def attention_weights(self, image_tokens: Tensor, corrupted_ids: Tensor) -> Tensor:
        """Diagnostic B×8×77×193 weights; scoring is QKᵀ/√64 before softmax."""
        queries: Tensor = self.ln_pre_t(self.text_tokens(corrupted_ids))
        keys: Tensor = self.ln_pre_i(image_tokens)
        weights: Tensor = self.cross_attn(queries, keys, keys, need_weights=True, average_attn_weights=False)[1]
        return weights


def cross_attention_scores(model: RetrievalModel, image_tokens: Tensor, corrupted_ids: Tensor) -> Tensor:
    """Expose actual pre-softmax attention scores for educational diagnostics."""
    queries: Tensor = model.ln_pre_t(model.text_tokens(corrupted_ids))
    keys: Tensor = model.ln_pre_i(image_tokens)
    projection: Tensor = model.cross_attn.in_proj_weight
    bias: Tensor = model.cross_attn.in_proj_bias
    q: Tensor = F.linear(queries, projection[:512], bias[:512]).reshape(queries.shape[0], -1, 8, 64).transpose(1, 2)
    k: Tensor = F.linear(keys, projection[512:1024], bias[512:1024]).reshape(keys.shape[0], -1, 8, 64).transpose(1, 2)
    return q @ k.transpose(-1, -2) / math.sqrt(64)
