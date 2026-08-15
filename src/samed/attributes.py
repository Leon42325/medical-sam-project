"""Object attributes used to explain SAM's medical segmentation performance.

Reference
---------
Y. Huang et al., "Segment anything model for medical images?",
*Medical Image Analysis* 92 (2024) 103061, Sec. 4.8
("Analysis of factors correlating to segmentation results").

The paper records five factors per anatomical structure -- size, aspect ratio,
intensity difference, boundary complexity and modality -- and correlates them
with DICE (Table 6, Figs. 13-15).  That analysis is the backbone of the present
project: hypotheses H1 and H2 are both statements about these attributes.  None
of it is released upstream, so it is implemented here from the paper text.

Boundary complexity is expressed through elliptic Fourier descriptors
(Kuhl & Giardina, 1982).  The descriptors themselves come from the ``pyefd``
package rather than being reimplemented; this module contributes only the
paper's iterative order-selection procedure on top.

A note on the paper's Eq. (2): as printed, the Fourier arguments read
``sin(T / (2 n pi t))`` -- the reciprocal of the standard elliptic Fourier
formulation ``sin(2 n pi t / T)``.  We treat this as a typesetting error and use
the standard Kuhl-Giardina form, which is what ``pyefd`` implements and what the
paper's own Fig. 12 (contours converging as the order grows) demonstrates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import pyefd

from samed.metrics import dice

__all__ = [
    "FourierOrder",
    "area",
    "aspect_ratio",
    "intensity_difference",
    "fourier_order",
    "largest_contour",
]


def _as_mask(mask) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {arr.shape}")
    return arr.astype(bool, copy=False)


def area(mask) -> int:
    """Pixel area of the mask.

    "The size of the anatomical structure was computed as the pixel-level area
    of the corresponding mask."
    """
    return int(_as_mask(mask).sum())


def aspect_ratio(mask) -> float:
    """Ratio of the short to the long side of the tight bounding box, in ``(0, 1]``.

    "we need to calculate the ratio (ranging from 0 to 1) between the short and
    long sides of its bounding box".
    """
    m = _as_mask(mask)
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        raise ValueError("mask is empty")
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    return min(width, height) / max(width, height)


def intensity_difference(image, mask, *, expand_ratio: float = 0.1, signed: bool = False) -> float:
    """Mean-intensity contrast between the structure and its surroundings.

    "The intensity difference was defined as the variation in mean intensity
    values between the structure and the surrounding area within an enlarged
    bounding box, excluding the structure itself.  Specifically, to accommodate
    the varied dimensions of targets, we dynamically expanded the box outward by
    a preset ratio of 0.1, instead of using fixed pixel values."

    AMBIGUITY: "expanded the box outward by a preset ratio of 0.1" does not say
    whether 0.1 is applied per side or to the total extent.  We extend each side
    outward by ``expand_ratio`` times the corresponding box dimension (so a
    100 px wide box gains 10 px on the left and 10 px on the right).  Because the
    quantity enters the analysis only through a rank correlation, the choice
    shifts values but is close to monotone in the alternative reading.

    A colour image is reduced to greyscale first, matching the paper's
    preprocessing, which normalises everything to single-channel 0-255.
    """
    m = _as_mask(mask)
    img = np.asarray(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if img.shape != m.shape:
        raise ValueError(f"image shape {img.shape} does not match mask shape {m.shape}")
    img = img.astype(np.float64, copy=False)

    ys, xs = np.nonzero(m)
    if ys.size == 0:
        raise ValueError("mask is empty")
    height, width = m.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad_x = expand_ratio * (x1 - x0 + 1)
    pad_y = expand_ratio * (y1 - y0 + 1)

    ex0 = max(0, int(np.floor(x0 - pad_x)))
    ex1 = min(width - 1, int(np.ceil(x1 + pad_x)))
    ey0 = max(0, int(np.floor(y0 - pad_y)))
    ey1 = min(height - 1, int(np.ceil(y1 + pad_y)))

    window = np.zeros_like(m)
    window[ey0 : ey1 + 1, ex0 : ex1 + 1] = True
    surround = window & ~m
    if not surround.any():
        raise ValueError("enlarged bounding box contains no background pixels")

    difference = float(img[m].mean() - img[surround].mean())
    return difference if signed else abs(difference)


def largest_contour(mask) -> np.ndarray:
    """Longest external contour of the mask as an ``(P, 2)`` array of ``(x, y)``.

    Objects that consist of several components (common for vessels or cells) are
    reduced to their largest component, so that boundary complexity describes one
    connected shape rather than the union of many.
    """
    m = _as_mask(mask)
    contours, _ = cv2.findContours(
        m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise ValueError("mask is empty; no contour found")
    largest = max(contours, key=lambda c: c.shape[0])
    return largest.reshape(-1, 2).astype(np.float64)


@dataclass(frozen=True)
class FourierOrder:
    """Outcome of the iterative order search of Sec. 4.8.

    ``value`` is the paper's ``F_final``; it is what enters the correlation
    analysis.  ``order`` and ``dice`` are kept so that the penalty term can be
    inspected, and ``stopped_by`` records which of the paper's two termination
    criteria fired -- or ``"max_order"`` when neither did, which flags a contour
    the descriptor could not fit within the cap.
    """

    value: float
    order: int
    dice: float
    stopped_by: Literal["dice_target", "no_improvement", "max_order", "degenerate"]


def fourier_order(
    mask,
    *,
    dice_target: float = 0.97,
    min_improvement: float = 1e-3,
    max_order: int = 200,
    penalty: float = 2.0,
    patience: int | None = 1,
) -> FourierOrder:
    """Boundary complexity as the elliptic Fourier order needed to fit the contour.

    "For different structures, we increased the FO from 1 and calculated the DICE
    between the decoded contour and the original contour at each step.  Then we
    set two ways to end the process: (1) DICE > 97.0%; (2) the difference in the
    DICE between order F_(a-1) and F_(a) is less than 0.1%.  Consequently, we
    record the FO (F_a) and DICE after termination.  Finally, we take
    F_final = F_a + n x 100 x (1 - DICE), n = 2 as the final optimized FO."

    ``max_order`` caps the search, which the paper motivates by the EFD otherwise
    "getting stuck in endless accumulation".

    Criterion (2) is unsound as specified
    -------------------------------------
    The DICE-versus-order curve of an elliptic Fourier fit is a *staircase*, not
    a monotone climb: a shape with roughly k-fold symmetry puts almost all of its
    energy near harmonics k-1 and k+1, so orders 2..k-2 add nothing measurable.
    Criterion (2) stops at the first plateau, which for such shapes is order 2 —
    long before the contour is fitted.  Measured on synthetic k-pointed stars
    (see ``tests/test_attributes.py``), criterion (2) fires at order 2 for k = 6,
    8 and 14, whereas DICE > 0.97 is first reached at orders 11, 13 and never.

    The consequence is that the published attribute is not an order at all: with
    F_a pinned near 2, ``F_final`` reduces to ``2 + 200 (1 - DICE)``, a low-order
    fit residual.  That is consistent with the range up to ~180 seen in the
    paper's Fig. 14, which no genuine harmonic order would reach.  The measure
    still ranks shapes by complexity monotonically, so the paper's qualitative
    conclusion is not necessarily wrong — but its scale and its interpretation
    are.

    ``patience`` therefore selects the variant:

    * ``patience=1`` reproduces the paper exactly (default, used for the
      reproduction results).
    * ``patience=None`` disables criterion (2) altogether, so the search runs to
      the DICE target or to ``max_order``.  This is the corrected variant used
      for the sensitivity analysis in the report.
    * ``patience=k`` requires k consecutive non-improving orders, which crosses
      plateaus shorter than k.
    """
    contour = largest_contour(mask)
    if contour.shape[0] < 5:
        # Too few boundary pixels for a meaningful harmonic decomposition.
        return FourierOrder(value=1.0, order=1, dice=1.0, stopped_by="degenerate")

    shape = _as_mask(mask).shape
    reference = _rasterise(contour, shape)
    locus = pyefd.calculate_dc_coefficients(contour)
    n_points = contour.shape[0]

    previous = -np.inf
    current = 0.0
    order = 1
    stalled = 0
    stopped_by: str = "max_order"

    for order in range(1, max_order + 1):
        coeffs = pyefd.elliptic_fourier_descriptors(contour, order=order, normalize=False)
        decoded = pyefd.reconstruct_contour(coeffs, locus=locus, num_points=n_points)
        current = dice(_rasterise(decoded, shape), reference)

        if current > dice_target:
            stopped_by = "dice_target"
            break
        if patience is not None and order > 1:
            stalled = stalled + 1 if (current - previous) < min_improvement else 0
            if stalled >= patience:
                stopped_by = "no_improvement"
                break
        previous = current

    value = order + penalty * 100.0 * (1.0 - current)
    return FourierOrder(value=float(value), order=int(order), dice=float(current), stopped_by=stopped_by)  # type: ignore[arg-type]


def _rasterise(contour: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Fill a closed ``(P, 2)`` ``(x, y)`` contour into a boolean mask."""
    canvas = np.zeros(shape, dtype=np.uint8)
    polygon = np.round(contour).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [polygon], color=1)
    return canvas.astype(bool)
