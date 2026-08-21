"""Adapter for SAM (Kirillov et al., 2023) and its box-prompt fine-tune MedSAM.

Both share the architecture in ``segment_anything``; they differ in weights, in
which prompts they accept, and in how many masks they emit.  Nothing here
reimplements either model - the upstream package is a dependency, and this file
only reshapes its API and adds the embedding cache.

The cache is the reason this wrapper exists.  ``SamPredictor.set_image`` runs the
image encoder and stashes the result on the predictor; the study needs that
result once per image but reuses it across six strategies, three jitter levels
and three seeds.  :meth:`SamWrapper.encode` extracts it, and :meth:`predict`
puts it back, which is the same trick the authors' own repository uses for its
box path (``pre_grey_rgb2D.py`` writes ``.npy`` embeddings that
``test_only_box.py`` reloads), extended here to every strategy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from samed.models.base import MaskSet, PromptableSegmenter, register
from samed.prompts import Prompt

__all__ = ["SamWrapper", "as_rgb_uint8"]


class SamWrapper(PromptableSegmenter):
    """Wraps ``segment_anything`` for one checkpoint."""

    def __init__(
        self,
        *,
        name: str,
        variant: str,
        checkpoint: str,
        device: str = "cuda",
        supports: frozenset[str] = frozenset({"points", "box", "everything"}),
        multimask_output: bool = True,
        lora: str | None = None,
        decoder: str | None = None,
    ) -> None:
        import torch
        from segment_anything import SamPredictor, sam_model_registry

        self.name = name
        self.supports = supports
        self.input_size = 1024
        self.multimask_output = multimask_output

        self._torch = torch
        self._device = device
        model = sam_model_registry[variant](checkpoint=checkpoint)
        if lora or decoder:
            model = _apply_adaptation(model, lora=lora, decoder=decoder, torch=torch)
        self._model = model.to(device).eval()
        self._predictor = SamPredictor(self._model)

    # ------------------------------------------------------------------ encode
    def encode(self, image: np.ndarray) -> dict[str, Any]:
        """Run the image encoder once and return the cacheable state.

        Stored as float16: the embedding is 256x64x64, so fp16 halves the cache
        to 2 MB per image, and the encoder's own output precision does not
        justify float32 on disk.
        """
        image = as_rgb_uint8(image)
        with self._torch.inference_mode():
            self._predictor.set_image(image)
            features = self._predictor.features.detach().cpu().numpy().astype(np.float16)

        cached = {
            "features": features,
            "original_size": np.asarray(self._predictor.original_size, dtype=np.int64),
            "input_size": np.asarray(self._predictor.input_size, dtype=np.int64),
        }
        self._predictor.reset_image()
        return cached

    def _restore(self, cached: dict[str, Any]) -> None:
        self._predictor.reset_image()
        features = self._torch.from_numpy(np.asarray(cached["features"]).astype(np.float32))
        self._predictor.features = features.to(self._device)
        self._predictor.original_size = tuple(int(v) for v in cached["original_size"])
        self._predictor.input_size = tuple(int(v) for v in cached["input_size"])
        self._predictor.is_image_set = True

    # ----------------------------------------------------------------- predict
    def predict(self, cached: dict[str, Any], prompt: Prompt) -> MaskSet:
        if not self.supports_strategy(prompt.strategy):
            raise ValueError(f"{self.name} does not support strategy {prompt.strategy}")

        self._restore(cached)
        with self._torch.inference_mode():
            masks, scores, _ = self._predictor.predict(
                point_coords=prompt.point_coords,
                point_labels=prompt.point_labels,
                box=None if prompt.box is None else prompt.box[None, :],
                multimask_output=self.multimask_output,
            )
        self._predictor.reset_image()
        return MaskSet(
            masks=np.asarray(masks, dtype=bool),
            scores=np.asarray(scores, dtype=np.float32),
            model=self.name,
            strategy=prompt.strategy,
        )

    # -------------------------------------------------------------- everything
    def everything(self, image: np.ndarray, points_per_side: int = 32) -> MaskSet:
        """Strategy S1, via the upstream automatic mask generator.

        ``points_per_side`` defaults to 32, the value the paper calls SAM's
        default ("In default, m is set to 32") and the setting its Table 5
        ablates from 8 to 256.
        """
        from segment_anything import SamAutomaticMaskGenerator

        generator = SamAutomaticMaskGenerator(self._model, points_per_side=points_per_side)
        with self._torch.inference_mode():
            records = generator.generate(as_rgb_uint8(image))

        if not records:
            height, width = image.shape[:2]
            return MaskSet(
                masks=np.zeros((0, height, width), dtype=bool),
                scores=np.zeros((0,), dtype=np.float32),
                model=self.name,
                strategy="S1",
            )
        return MaskSet(
            masks=np.stack([r["segmentation"] for r in records]).astype(bool),
            scores=np.asarray([r["predicted_iou"] for r in records], dtype=np.float32),
            model=self.name,
            strategy="S1",
        )


def _apply_adaptation(model, *, lora: str | None, decoder: str | None, torch):
    """Load what a fine-tuning arm learned back onto a pretrained SAM.

    LoRA adapters are merged into the base weights rather than left as separate
    modules: merging costs nothing at load time, removes the per-layer overhead
    at inference, and leaves a plain SAM that every existing code path - the
    predictor, the embedding cache, the automatic mask generator - handles
    without knowing an adapter was ever involved.
    """
    if lora:
        from peft import PeftModel

        wrapped = PeftModel.from_pretrained(model.image_encoder, lora)
        model.image_encoder = wrapped.merge_and_unload()
    if decoder:
        state = torch.load(decoder, map_location="cpu", weights_only=True)
        missing, unexpected = model.mask_decoder.load_state_dict(state, strict=True)
        assert not missing and not unexpected
    return model


def as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """SAM expects HWC uint8 RGB; the preprocessed data is single-channel 0-255.

    Shared with the fine-tuning path deliberately. Training and inference must
    agree on this conversion, and a second copy is a second chance to disagree.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


@register("sam_vit_b")
def _sam_vit_b(checkpoint: str, device: str = "cuda", *, lora: str | None = None,
            decoder: str | None = None, name: str = "sam_vit_b") -> SamWrapper:
    return SamWrapper(name=name, variant="vit_b", checkpoint=checkpoint,
                      device=device, lora=lora, decoder=decoder)


@register("sam_vit_h")
def _sam_vit_h(checkpoint: str, device: str = "cuda", *, lora: str | None = None,
            decoder: str | None = None, name: str = "sam_vit_h") -> SamWrapper:
    return SamWrapper(name=name, variant="vit_h", checkpoint=checkpoint,
                      device=device, lora=lora, decoder=decoder)


@register("medsam")
def _medsam(checkpoint: str, device: str = "cuda", *, lora: str | None = None,
            decoder: str | None = None, name: str = "medsam") -> SamWrapper:
    """MedSAM: the same ViT-B architecture, fine-tuned for box prompts only.

    ``supports`` excludes points deliberately.  MedSAM was trained exclusively
    with box prompts, and its own repository exposes no point interface; feeding
    it points would produce numbers that say nothing about domain adaptation.
    Strategies S2-S4 are therefore reported as N/A rather than approximated.
    ``multimask_output`` is False to match how the authors run it.
    """
    return SamWrapper(
        name=name,
        variant="vit_b",
        checkpoint=checkpoint,
        device=device,
        supports=frozenset({"box"}),
        multimask_output=False,
        lora=lora,
        decoder=decoder,
    )
