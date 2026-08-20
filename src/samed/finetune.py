"""Adapting SAM to medical images, and where that adaptation is allowed to happen.

Huang et al. fine-tune SAM in Sec. 4.12 and report +4.39 DICE for ViT-B. They
also state the constraint that produced that number:

    "We fixed the image encoder to minimize computation costs and also kept the
    prompt encoder frozen [...] Thus, only the parameters in the mask decoder
    were adjusted during finetuning."

The image encoder was frozen for compute, not for a reason about learning. That
makes the reported gain a lower bound of a particular kind: a mask decoder can
only reweight what the encoder already represents. If SAM fails on a
low-contrast pancreas because its features do not separate the organ from
surrounding tissue, no decoder recovers it.

Low-rank adaptation removes the compute constraint, so the question becomes
answerable: **is SAM's weakness on medical images a representation problem or a
readout problem?** The arms below differ only in where the gradient is allowed
to flow.

Parameter counts make the comparison sharper rather than blunter. For ViT-B the
mask decoder holds 4.06M parameters and LoRA of rank 8 over the attention
projections holds 0.455M - so the decoder-only arm trains nine times as many.
If the smaller intervention on the encoder wins, the result cannot be explained
by capacity.

LoRA itself is not implemented here; ``peft`` provides it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

__all__ = ["Arm", "FineTuneConfig", "prepare_model", "trainable_parameters", "segmentation_loss"]

Arm = Literal["decoder", "lora_encoder", "lora_encoder_decoder"]

#: Which linear projections LoRA is attached to. Attention only by default,
#: which is where the SAMed line of work puts it and which keeps the adapter an
#: order of magnitude below the decoder it is being compared against.
DEFAULT_LORA_TARGETS = ("qkv", "proj")


@dataclass(frozen=True)
class FineTuneConfig:
    """One arm of the fine-tuning experiment.

    Defaults follow the paper where it states a value - box prompts, frozen
    prompt encoder, lr 1e-4, batch 2, 20 epochs - so that the ``decoder`` arm is
    a replication and the LoRA arms differ from it in exactly one respect.
    """

    arm: Arm = "decoder"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_targets: Sequence[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))
    learning_rate: float = 1e-4
    batch_size: int = 2
    epochs: int = 20
    seed: int = 0

    @property
    def trains_encoder(self) -> bool:
        return self.arm in ("lora_encoder", "lora_encoder_decoder")

    @property
    def trains_decoder(self) -> bool:
        return self.arm in ("decoder", "lora_encoder_decoder")

    @property
    def can_use_cached_embeddings(self) -> bool:
        """Whether training can read the embedding cache instead of the images.

        Only when the encoder is frozen. That is not a detail: with a frozen
        encoder the whole forward pass reduces to a decode from a stored
        256x64x64 tensor, which is two orders of magnitude cheaper than
        re-encoding a 1024x1024 image every step. The decoder arm therefore
        trains in minutes and the LoRA arms in hours, and the gap is inherent
        rather than an implementation shortcoming.
        """
        return not self.trains_encoder


def prepare_model(sam, config: FineTuneConfig):
    """Freeze everything, then re-enable exactly what this arm adapts.

    The prompt encoder is never trained, in any arm: the paper freezes it "because
    of its powerful capacity for encoding box positional information", and keeping
    that fixed is what makes the arms comparable.
    """
    from peft import LoraConfig, get_peft_model

    for parameter in sam.parameters():
        parameter.requires_grad_(False)

    if config.trains_encoder:
        sam.image_encoder = get_peft_model(
            sam.image_encoder,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.lora_targets),
                bias="none",
            ),
        )

    if config.trains_decoder:
        for parameter in sam.mask_decoder.parameters():
            parameter.requires_grad_(True)

    return sam


def trainable_parameters(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def segmentation_loss(logits, targets, *, dice_weight: float = 1.0, bce_weight: float = 1.0):
    """Dice plus binary cross-entropy on the raw mask logits.

    The combination MedSAM trains with, and the paper says its fine-tuning was
    "inspired by" MedSAM. Dice alone is unstable when a prediction starts empty,
    which happens often on the low-contrast targets this experiment is about;
    BCE supplies a gradient there.
    """
    import torch
    import torch.nn.functional as functional

    targets = targets.to(dtype=logits.dtype)
    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * targets).sum(dim=(-2, -1))
    total = probabilities.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * intersection + 1.0) / (total + 1.0)

    bce = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return dice_weight * dice.mean() + bce_weight * bce.mean(dim=(-2, -1)).mean()


def decode(sam, embeddings, boxes, input_size, original_size):
    """Mask logits at the original resolution, from embeddings and box prompts.

    Split out from the encoder so that both training paths share it. With the
    encoder frozen the embeddings come from the cache and this is the entire
    forward pass; with LoRA attached they come from ``sam.image_encoder`` and
    this is only the tail.

    Box prompts because the paper fine-tunes with box prompts, and the prompt
    encoder stays in ``no_grad`` because it is frozen in every arm - running it
    under gradient would only build a graph nothing can use.
    """
    import torch

    with torch.no_grad():
        sparse, dense = sam.prompt_encoder(points=None, boxes=boxes, masks=None)

    low_resolution, _ = sam.mask_decoder(
        image_embeddings=embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    # postprocess_masks removes the padding SAM added and resizes back, so the
    # loss is taken against the annotation as drawn rather than against a
    # letterboxed copy of it.
    return sam.postprocess_masks(low_resolution, input_size, original_size)
