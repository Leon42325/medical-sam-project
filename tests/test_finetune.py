"""Tests for the fine-tuning arms and their loss.

Built on a miniature SAM rather than ViT-B: the surgery under test is about
which parameters carry gradient, which does not depend on depth, and a two-block
encoder keeps the suite fast enough to stay in the normal edit-run loop.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="fine-tuning operates on torch models")
pytest.importorskip("peft", reason="LoRA comes from peft")

from segment_anything.build_sam import _build_sam  # noqa: E402

from samed.finetune import (  # noqa: E402
    FineTuneConfig,
    prepare_model,
    segmentation_loss,
    trainable_parameters,
)


def tiny_sam():
    return _build_sam(
        encoder_embed_dim=48, encoder_depth=2, encoder_num_heads=2,
        encoder_global_attn_indexes=[], checkpoint=None,
    )


@pytest.mark.parametrize("arm", ["decoder", "lora_encoder", "lora_encoder_decoder"])
def test_every_arm_trains_something(arm):
    model = prepare_model(tiny_sam(), FineTuneConfig(arm=arm))
    assert trainable_parameters(model) > 0


def test_the_prompt_encoder_is_never_trained():
    """The paper freezes it in every configuration; keeping it fixed is what
    makes the arms differ in exactly one respect."""
    for arm in ("decoder", "lora_encoder", "lora_encoder_decoder"):
        model = prepare_model(tiny_sam(), FineTuneConfig(arm=arm))
        assert trainable_parameters(model.prompt_encoder) == 0


def test_the_decoder_arm_leaves_the_encoder_frozen():
    model = prepare_model(tiny_sam(), FineTuneConfig(arm="decoder"))
    assert trainable_parameters(model.image_encoder) == 0
    assert trainable_parameters(model.mask_decoder) > 0


def test_the_lora_arm_leaves_the_decoder_frozen():
    model = prepare_model(tiny_sam(), FineTuneConfig(arm="lora_encoder"))
    assert trainable_parameters(model.mask_decoder) == 0
    assert trainable_parameters(model.image_encoder) > 0


def test_lora_touches_only_adapter_parameters():
    """The base weights must stay frozen, or the arm is no longer low-rank."""
    model = prepare_model(tiny_sam(), FineTuneConfig(arm="lora_encoder"))
    trainable = [n for n, p in model.image_encoder.named_parameters() if p.requires_grad]
    assert trainable
    assert all("lora" in name for name in trainable)


def test_rank_controls_the_adapter_size():
    small = prepare_model(tiny_sam(), FineTuneConfig(arm="lora_encoder", lora_rank=2))
    large = prepare_model(tiny_sam(), FineTuneConfig(arm="lora_encoder", lora_rank=16))
    assert trainable_parameters(large) > 4 * trainable_parameters(small)


def test_wider_targets_add_parameters():
    attention = prepare_model(
        tiny_sam(), FineTuneConfig(arm="lora_encoder", lora_targets=["qkv"]))
    everything = prepare_model(
        tiny_sam(),
        FineTuneConfig(arm="lora_encoder", lora_targets=["qkv", "proj", "lin1", "lin2"]))
    assert trainable_parameters(everything) > trainable_parameters(attention)


def test_only_a_frozen_encoder_permits_the_embedding_cache():
    """Training the encoder invalidates cached embeddings by construction; the
    arms differ in cost by two orders of magnitude for exactly this reason."""
    assert FineTuneConfig(arm="decoder").can_use_cached_embeddings
    assert not FineTuneConfig(arm="lora_encoder").can_use_cached_embeddings
    assert not FineTuneConfig(arm="lora_encoder_decoder").can_use_cached_embeddings


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #


def test_loss_is_near_zero_for_a_confident_correct_prediction():
    target = torch.zeros(1, 32, 32)
    target[:, 8:24, 8:24] = 1
    logits = torch.where(target > 0, 12.0, -12.0)
    assert segmentation_loss(logits, target).item() < 0.02


def test_loss_is_large_for_a_confident_wrong_prediction():
    target = torch.zeros(1, 32, 32)
    target[:, 8:24, 8:24] = 1
    logits = torch.where(target > 0, -12.0, 12.0)
    assert segmentation_loss(logits, target).item() > 5


def test_loss_decreases_as_a_prediction_improves():
    target = torch.zeros(1, 32, 32)
    target[:, 8:24, 8:24] = 1
    losses = [
        segmentation_loss(torch.where(target > 0, scale, -scale), target).item()
        for scale in (0.1, 1.0, 4.0)
    ]
    assert losses == sorted(losses, reverse=True)


def test_bce_keeps_a_gradient_when_the_prediction_is_empty():
    """Dice alone saturates on an empty prediction, which is the common failure
    on the low-contrast targets this experiment exists to probe."""
    target = torch.zeros(1, 16, 16)
    target[:, 4:12, 4:12] = 1
    logits = torch.full((1, 16, 16), -20.0, requires_grad=True)

    segmentation_loss(logits, target, dice_weight=1.0, bce_weight=0.0).backward()
    dice_only = logits.grad.abs().sum().item()

    logits.grad = None
    segmentation_loss(logits, target).backward()
    with_bce = logits.grad.abs().sum().item()

    assert with_bce > 10 * dice_only


def test_loss_accepts_a_batch():
    target = torch.zeros(4, 1, 24, 24)
    target[:, :, 6:18, 6:18] = 1
    logits = torch.where(target > 0, 8.0, -8.0)
    assert segmentation_loss(logits, target).item() < 0.05
