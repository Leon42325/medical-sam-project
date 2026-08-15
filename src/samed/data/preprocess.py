"""Preprocessing protocol of Huang et al. (2024), Sec. 2.2.

The study assumes every dataset has been reduced to single-channel 8-bit PNG
slices with integer-valued label maps.  That normalisation is what makes 18
modalities with wildly different value ranges (MRI 0-800, CT -2000..2000,
others already 0-255) comparable at all, so it has to be reproduced faithfully
rather than approximated.

Quoted rules, for 3D volumes:

  "(1) Extract slices along the main viewing plane since it has higher
  resolution. (2) Retain slices with the sum of the pixel values of their labels
  greater than 50 for any 3D image and label volume. (3) Normalize the image
  intensities by min-max normalization: I_n = 255 * (I - I_min) / (I_max -
  I_min), limiting the range to (0, 255) [...] (4) Save images and labels in PNG
  format."

and for 2D images:

  "(1) Retain images with the sum of the pixel values of their labels greater
  than 50. (2) Reset the pixel value of the labels according to the object
  category [...] (3) Convert the format of images and labels from BMP, JPG, TIF,
  etc. to PNG for achieving consistent data loading."
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = ["min_max_normalise", "slice_label_areas", "select_labelled_slices"]

#: The paper's minimum label size for a slice to be kept (Sec. 2.2).
MIN_LABEL_AREA = 50


#: Percentiles the intensity range is computed over, instead of the raw extremes.
#: See the AMBIGUITY note in :func:`min_max_normalise`.
DEFAULT_CLIP_PERCENTILES = (0.5, 99.5)


def min_max_normalise(
    array,
    *,
    scope: Literal["volume", "slice"] = "volume",
    axis: int = 0,
    clip_percentiles: tuple[float, float] | None = DEFAULT_CLIP_PERCENTILES,
) -> np.ndarray:
    """Min-max normalise intensities to 0-255 and return ``uint8``.

    ``I_n = 255 * (I - I_min) / (I_max - I_min)`` (Sec. 2.2, step 3).

    AMBIGUITY: taken literally, ``I_min`` and ``I_max`` are the raw extremes, and
    that is not usable. Medical volumes carry sentinel and artefact values far
    outside the physical range: CHAOS patient CT/19 spans -1000 to 49944, where
    no tissue exceeds ~1500 HU. Normalising over that range compresses every real
    structure into the bottom 3% of the output, and the exported slice is, to the
    eye and to a segmentation model, black. The paper cannot have done this - its
    figures show normal contrast - but it does not say what it did instead.

    We therefore take the range over percentiles by default and clip beyond them,
    which is the standard remedy and leaves images with sentinel values usable.
    ``clip_percentiles=None`` restores the literal reading, for the sensitivity
    check reported alongside the results.

    AMBIGUITY: for a 3D volume the paper writes that "I means the original
    normalized image, I_n represents the normalized image.  I_min and I_max are
    the minimum and maximum intensity of the original image", without saying
    whether "the original image" is the volume or the individual slice.  The two
    differ substantially: per-slice normalisation destroys the intensity
    relationship between slices and would, for instance, make an all-soft-tissue
    CT slice look like one containing bone.  We default to ``scope="volume"``,
    which preserves it; ``scope="slice"`` is available for a sensitivity check.

    A constant array maps to all zeros rather than dividing by zero.
    """
    data = np.asarray(array, dtype=np.float64)

    if scope == "volume":
        if clip_percentiles is None:
            low, high = data.min(), data.max()
        else:
            low, high = np.percentile(data, clip_percentiles)
            data = np.clip(data, low, high)
        scale = high - low
        scaled = np.zeros_like(data) if scale == 0 else (data - low) / scale
    elif scope == "slice":
        moved = np.moveaxis(data, axis, 0)
        flat_axes = tuple(range(1, moved.ndim))
        if clip_percentiles is None:
            low = moved.min(axis=flat_axes, keepdims=True)
            high = moved.max(axis=flat_axes, keepdims=True)
        else:
            low, high = (
                np.percentile(moved, p, axis=flat_axes, keepdims=True)
                for p in clip_percentiles
            )
            moved = np.clip(moved, low, high)
        scale = high - low
        scaled = np.divide(
            moved - low, scale, out=np.zeros_like(moved), where=scale != 0
        )
        scaled = np.moveaxis(scaled, 0, axis)
    else:  # pragma: no cover - guarded by typing
        raise ValueError(f"unknown scope {scope!r}")

    return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)


def slice_label_areas(labels, *, axis: int = 0, label_value: int | None = None) -> np.ndarray:
    """Number of labelled pixels per slice along ``axis``.

    AMBIGUITY: the paper thresholds "the sum of the pixel values of their
    labels".  Read literally that is the sum of the *category codes*, under which
    a single pixel of a class encoded as 85 would already exceed the threshold of
    50 - which cannot be the intent, since the stated purpose is "to ensure that
    each slice has the corresponding correct label".  We therefore count labelled
    pixels.  Pass ``label_value`` to count one category rather than all.
    """
    data = np.asarray(labels)
    mask = data != 0 if label_value is None else data == label_value
    moved = np.moveaxis(mask, axis, 0)
    return moved.reshape(moved.shape[0], -1).sum(axis=1)


def select_labelled_slices(
    labels,
    *,
    axis: int = 0,
    min_area: int = MIN_LABEL_AREA,
    label_value: int | None = None,
) -> np.ndarray:
    """Indices of slices whose label is large enough to keep (Sec. 2.2, step 2).

    The threshold is strict ("greater than 50"), and it is what removes the
    partial-volume slices at the top and bottom of an organ where the annotation
    degenerates to a few pixels.
    """
    areas = slice_label_areas(labels, axis=axis, label_value=label_value)
    return np.flatnonzero(areas > min_area)


def read_pixels(path) -> np.ndarray:
    """Pixel data of one slice, in the source's own intensity units.

    DICOM stores raw stored values; ``RescaleSlope`` and ``RescaleIntercept``
    convert them to the physical scale (Hounsfield units for CT). Applying the
    rescale before normalisation is what makes a min-max over a CT volume mean
    the same thing across patients - without it, the normalisation is over an
    arbitrary vendor offset.
    """
    from pathlib import Path

    file = Path(path)
    if file.suffix.lower() in {".dcm", ".ima"}:
        import pydicom

        dataset = pydicom.dcmread(str(file))
        pixels = dataset.pixel_array.astype(np.float64)
        slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
        return pixels * slope + intercept

    import cv2

    image = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"could not read {file}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float64)
