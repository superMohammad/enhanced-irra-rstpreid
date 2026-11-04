"""Behavioral regression tests; synthetic fixtures are never benchmark evidence."""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from enhanced_irra.data import ImageRecord, read_records, verify_training_isolation
from enhanced_irra.losses import identity_distribution, mlm_loss, sdm_loss
from enhanced_irra.masking import MASK_ID, VOCAB_SIZE, frequency_weights, mask_caption, tokenizer_connector
from enhanced_irra.metrics import retrieval_metrics
from enhanced_irra.model import RetrievalModel, cross_attention_scores
from enhanced_irra.vendor.clip_model import CLIP


def test_sdm_multi_positive_normalization_and_gradient() -> None:
    identities = torch.tensor([1, 1, 2, 2, 2])
    targets = identity_distribution(identities)
    assert torch.allclose(targets.sum(1), torch.ones(5))
    assert torch.allclose(targets[0], torch.tensor([0.5, 0.5, 0, 0, 0]))
    assert torch.allclose(targets[2], torch.tensor([0, 0, 1/3, 1/3, 1/3]))
    images = torch.randn(5, 16, requires_grad=True)
    texts = torch.randn(5, 16, requires_grad=True)
    loss = sdm_loss(images, texts, identities, 0.02)
    loss.backward()
    assert torch.isfinite(loss)
    assert images.grad is not None and images.grad.abs().sum() > 0
    assert texts.grad is not None and texts.grad.abs().sum() > 0


def test_sdm_kl_direction_penalizes_negative_probability() -> None:
    ids = torch.tensor([0, 1])
    features = torch.eye(2)
    aligned = sdm_loss(features, features, ids, 0.02)
    swapped = sdm_loss(features, features.flip(0), ids, 0.02)
    assert aligned < 1e-5
    assert swapped > 30


@pytest.mark.parametrize("strategy", ["random", "attribute", "phrase"])
def test_masking_preserves_clean_original_targets_and_determinism(strategy: str) -> None:
    tokenizer = tokenizer_connector()
    caption = tokenizer.encode("A man wearing a black leather jacket and turquoise sneakers.")
    original = tuple(caption.token_ids)
    result = mask_caption(caption, strategy, 0.3, 17)  # type: ignore[arg-type]
    repeated = mask_caption(caption, strategy, 0.3, 17)  # type: ignore[arg-type]
    assert caption.token_ids == original
    assert result.clean.tolist() == list(original)
    assert result.clean.data_ptr() != result.corrupted.data_ptr()
    selected = result.targets.ne(-100)
    assert torch.equal(result.targets[selected], result.clean[selected])
    assert torch.equal(result.clean[~selected], result.corrupted[~selected])
    assert torch.equal(result.corrupted, repeated.corrupted)
    assert not selected[0] and not selected[result.clean.argmax()]


def test_phrase_masks_complete_occurrence_and_subwords() -> None:
    caption = tokenizer_connector().encode("A turquoise backpack beside a black leather jacket.")
    result = mask_caption(caption, "phrase", 0.15, 3)
    assert caption.phrases
    for phrase in caption.phrases:
        overlap = set(phrase) & set(result.selected)
        assert not overlap or overlap == set(phrase)
    assert result.stopword_selected == 0
    assert result.selected_phrase_count >= 1


def test_semantic_policies_never_substitute_stopwords_when_no_candidates() -> None:
    caption = tokenizer_connector().encode("He is walking over there.")
    for strategy in ("attribute", "phrase"):
        result = mask_caption(caption, strategy, 0.15, 2)
        assert result.selected == ()
        assert result.targets.eq(-100).all()


def test_truncated_phrase_not_partially_selected() -> None:
    caption = tokenizer_connector().encode("walking " * 74 + "turquoise backpack")
    assert caption.truncated
    assert not caption.phrases
    assert not mask_caption(caption, "phrase", 0.15, 7).selected


def test_frequency_fit_rejects_validation_and_is_bounded() -> None:
    tokenizer = tokenizer_connector()
    train = ImageRecord(0, Path("unused"), ("black shirt black shirt", "red shoes"), "train", 1)
    val = ImageRecord(1, Path("unused"), ("yellow backpack", "white scarf"), "val", 1)
    with pytest.raises(ValueError, match="training records"):
        frequency_weights((train, val), tokenizer, 2.0)
    weights = frequency_weights((train,), tokenizer, 2.0)
    assert weights.shape == (VOCAB_SIZE,)
    assert 0.05 <= weights.min() <= weights.max() <= 1


def test_mlm_supervises_unchanged_selected_tokens_and_ignores_others() -> None:
    logits = torch.zeros(1, 3, 4, requires_grad=True)
    targets = torch.tensor([[-100, 1, -100]])
    loss = mlm_loss(logits, targets, torch.ones(4))
    loss.backward()
    assert loss.item() == pytest.approx(1.38629436)
    assert logits.grad is not None
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 1, 1] < 0
    assert logits.grad[0, 2].abs().sum() == 0


def test_empty_mlm_has_finite_differentiable_zero() -> None:
    logits = torch.randn(1, 3, 4, requires_grad=True)
    loss = mlm_loss(logits, torch.full((1, 3), -100), torch.ones(4))
    loss.backward()
    assert loss == 0 and logits.grad is not None and logits.grad.eq(0).all()


def test_metrics_identity_relevance_full_gallery_ap_and_inp() -> None:
    # Identity 7 occurs at ranks 1, 3, 4. AP=(1+2/3+3/4)/3; INP=3/4.
    result = retrieval_metrics(torch.tensor([[5., 4., 3., 2., 1.]]), torch.tensor([7]), torch.tensor([7, 8, 7, 7, 8]))
    assert result.rank1 == 100
    assert result.rank5 == 100
    assert result.mAP == pytest.approx(100 * (1 + 2/3 + 3/4) / 3)
    assert result.mINP == 75
    with pytest.raises(ValueError, match="relevant"):
        retrieval_metrics(torch.ones(1, 2), torch.tensor([9]), torch.tensor([1, 2]))


def test_equal_scores_have_stable_gallery_order() -> None:
    result = retrieval_metrics(torch.ones(1, 2), torch.tensor([1]), torch.tensor([0, 1]))
    assert result.rank1 == 0 and result.mAP == 50 and result.mINP == 50


def test_duplicate_rgb_image_rejected_before_training(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.png"
    Image.new("RGB", (4, 4), "black").save(first)
    Image.new("RGB", (4, 4), "black").save(second)
    records = (ImageRecord(1, first, ("a", "b"), "train", 1), ImageRecord(2, second, ("c", "d"), "val", 1))
    with pytest.raises(ValueError, match="leakage"):
        verify_training_isolation(records[:1], records[1:])


def test_unknown_split_does_not_become_validation(tmp_path: Path) -> None:
    (tmp_path / "imgs").mkdir()
    payload = [{"id": 0, "img_path": "0000_c1_0000.jpg", "captions": ["a", "b"], "split": "typo"}]
    (tmp_path / "data_captions.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unknown split"):
        read_records(tmp_path)


def test_actual_irr_forward_backward_clean_inputs_and_attention() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(19)
    # Deliberately small random fixture: verifies wiring, never retrieval quality.
    backbone = CLIP(embed_dim=512, image_resolution=(32, 16), vision_layers=1, vision_width=64, vision_patch_size=16, stride_size=16, context_length=77, vocab_size=VOCAB_SIZE, transformer_width=64, transformer_heads=1, transformer_layers=1)
    model = RetrievalModel(backbone, 2, 1).eval()
    tokenizer = tokenizer_connector()
    samples = [mask_caption(tokenizer.encode(c), "phrase", 0.3, 17) for c in ("A red jacket and black shoes.", "A white shirt and blue trousers.")]
    clean = torch.stack([s.clean for s in samples])
    corrupt = torch.stack([s.corrupted for s in samples])
    clean_copy = clean.clone()
    features = model(torch.randn(2, 3, 32, 16), clean)
    assert features.images.shape == features.texts.shape == (2, 512)
    assert features.image_tokens.shape == (2, 3, 512)
    assert torch.equal(features.texts, model.encode_text(clean_copy))
    logits = model.predict_masked(features.image_tokens, corrupt)
    assert logits.shape == (2, 77, VOCAB_SIZE)
    assert torch.equal(clean, clean_copy)
    scores = cross_attention_scores(model, features.image_tokens, corrupt)
    weights = model.attention_weights(features.image_tokens, corrupt)
    assert weights.shape == (2, 8, 77, 3)
    assert torch.allclose(scores.softmax(-1), weights, atol=1e-6)
    targets = torch.stack([s.targets for s in samples])
    loss = mlm_loss(logits, targets, torch.ones(VOCAB_SIZE))
    loss.backward()
    assert model.backbone.visual.conv1.weight.grad is not None
    assert model.backbone.visual.conv1.weight.grad.abs().sum() > 0
    assert model.backbone.token_embedding.weight.grad is not None
    assert model.backbone.token_embedding.weight.grad.abs().sum() > 0
    assert model.cross_attn.in_proj_weight.grad is not None
    assert torch.isfinite(loss)
    assert clean.eq(MASK_ID).sum() == 0
