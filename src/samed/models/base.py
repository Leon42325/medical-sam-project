"""The common interface every segmenter is driven through.

Design constraints, in order of importance:

1. **No model code lives here.** SAM, SAM 2, MedSAM, SAM-Med2D and MedSAM2 are
   consumed as upstream packages; this layer only adapts their APIs to one
   signature so the evaluation harness does not branch per model.

2. **Encoding is separated from prompting.** The image encoder dominates the
   cost (Table 4: 0.13 s for ViT-B, 0.47 s for ViT-H per image, against ~0.01 s
   for prompt encoding and mask decoding), and the study runs six strategies,
   three jitter levels and three seeds over the same images.  Encoding once and
   caching the result turns an otherwise ~50x redundant workload into a single
   pass - the difference between fitting in a shared GPU quota and not.

3. **All masks are returned, never one.** SAM emits several masks per prompt.
   The paper keeps whichever scores best against the ground truth, which uses
   the ground truth at inference time and therefore reports an unattainable
   upper bound (see PLAN.md Sec. 9.1).  Quantifying that requires the model's
   own quality estimate alongside every mask, so :class:`MaskSet` carries both
   and selection is deferred to :mod:`samed.selection`.

`torch` is imported lazily inside the concrete wrappers, so this module and the
whole analysis layer stay importable - and testable - without a GPU stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from samed.prompts import Prompt

__all__ = ["MaskSet", "PromptableSegmenter", "register", "create", "available",
           "encoder_identity"]


def encoder_identity(key: str, lora: str | None = None) -> str:
    """What the image encoder of a run actually is.

    Stored in every embedding cache and checked before the cache is used.
    Without it, a run whose encoder has been adapted silently reads
    embeddings produced by the unadapted one - the prediction path never
    runs the encoder at all - and the adaptation has exactly no effect on
    the results while the job reports success. That happened once, to the
    LoRA arms, and the only visible symptom was that two models scored
    identically to three decimals.

    The decoder is deliberately not part of the identity: it runs after the
    cache and a cached embedding stays valid across decoder changes.
    """
    from pathlib import Path

    return f"{key}+lora:{Path(lora).name}" if lora else key


@dataclass(frozen=True)
class MaskSet:
    """Every mask a model produced for one prompt, with its own quality scores.

    ``masks`` is ``(N, H, W)`` boolean at the original image resolution;
    ``scores`` is ``(N,)`` and holds the model's predicted IoU, which is the only
    selection signal available at deployment time.
    """

    masks: np.ndarray
    scores: np.ndarray
    model: str
    strategy: str

    def __post_init__(self) -> None:
        if self.masks.ndim != 3:
            raise ValueError(f"masks must be (N, H, W), got shape {self.masks.shape}")
        if self.scores.shape != (self.masks.shape[0],):
            raise ValueError(
                f"expected {self.masks.shape[0]} scores, got {self.scores.shape}"
            )

    def __len__(self) -> int:
        return int(self.masks.shape[0])


class PromptableSegmenter(ABC):
    """A model that turns an image plus a prompt into candidate masks."""

    #: Identifier used in results tables and in the registry.
    name: str

    #: Which of ``{"points", "box", "everything"}`` the model accepts.  MedSAM,
    #: for instance, is box-only, so strategies S2-S4 are reported as N/A for it
    #: rather than silently approximated.
    supports: frozenset[str]

    #: Side length the model resizes its input to.  Recorded because it is a
    #: confounder: SAM-Med2D runs at 256 against SAM's 1024, so a raw comparison
    #: mixes domain adaptation with a 16-fold reduction in input pixels.
    input_size: int

    @abstractmethod
    def encode(self, image: np.ndarray) -> dict[str, Any]:
        """Run the image encoder and return a cacheable, serialisable result.

        The value must survive a round trip through ``np.savez`` so it can be
        written once and reused by every later stage.
        """

    @abstractmethod
    def predict(self, cached: dict[str, Any], prompt: Prompt) -> MaskSet:
        """Decode masks for one prompt against a previously encoded image."""

    def everything(self, image: np.ndarray, points_per_side: int = 32) -> MaskSet:
        """Automatic mode (strategy S1).

        Not cacheable in the same way as :meth:`predict`, because the grid
        sampling and the non-maximum suppression that follow it depend on the
        whole image rather than on one prompt.
        """
        raise NotImplementedError(f"{self.name} does not support everything mode")

    def supports_strategy(self, strategy: str) -> bool:
        """Whether a strategy from S1-S6 is applicable to this model."""
        from samed.prompts import STRATEGY_SPEC

        if strategy == "S1":
            return "everything" in self.supports
        spec = STRATEGY_SPEC[strategy]
        if spec["box"] and "box" not in self.supports:
            return False
        if (spec["n_pos"] or spec["n_neg"]) and "points" not in self.supports:
            return False
        return True


_REGISTRY: dict[str, Callable[..., PromptableSegmenter]] = {}


def register(name: str) -> Callable[[Callable[..., PromptableSegmenter]], Callable[..., PromptableSegmenter]]:
    """Register a factory so experiments can name models in a config file."""

    def decorator(factory: Callable[..., PromptableSegmenter]) -> Callable[..., PromptableSegmenter]:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def create(key: str, **kwargs: Any) -> PromptableSegmenter:
    """Build a registered model.

    The parameter is ``key``, not ``name``: the registry key selects the
    architecture and weights, while ``name`` is what a run is recorded as, and a
    fine-tuning arm needs a different one from the checkpoint it started from.
    Conflating the two made ``create`` reject its own arguments.
    """
    if key not in _REGISTRY:
        raise KeyError(f"unknown model {key!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
