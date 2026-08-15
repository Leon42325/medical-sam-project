"""Segmentation metrics.

The three the paper reports (Sec. 4.2): DICE, JAC (IoU) and Hausdorff distance
in pixels. The boundary-aware metrics we add as an evaluation improvement
(normalised surface Dice, Boundary IoU) build on :func:`surface_points` here.

Empty-mask convention: two empty masks are identical (DICE 1, JAC 1, HD 0), but
an empty prediction against a non-empty reference is a total failure, which for
a distance metric means ``inf``. Returning ``inf`` rather than a large finite
number keeps the failure visible in aggregation instead of quietly averaging it
away - complete misses are common under the single-point strategies and the
paper's own HD values (up to ~1977 px for Lung-Xray under S1) show what happens
when they are folded into a mean.
"""

from __future__ import annotations

import numpy as np

__all__ = ["dice", "jaccard", "hausdorff", "surface_points"]


def dice(a, b) -> float:
    """Dice similarity coefficient between two binary masks, in ``[0, 1]``.

    Two empty masks are defined as perfectly similar (1.0), which keeps the
    metric well-behaved when a predicted mask and its reference are both empty.
    """
    x = np.asarray(a).astype(bool, copy=False)
    y = np.asarray(b).astype(bool, copy=False)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    total = int(x.sum()) + int(y.sum())
    if total == 0:
        return 1.0
    return 2.0 * float(np.logical_and(x, y).sum()) / total


def jaccard(a, b) -> float:
    """Jaccard similarity (IoU) between two binary masks, in ``[0, 1]``."""
    x = np.asarray(a).astype(bool, copy=False)
    y = np.asarray(b).astype(bool, copy=False)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    union = int(np.logical_or(x, y).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(x, y).sum()) / union


def surface_points(mask) -> np.ndarray:
    """Coordinates of the mask's boundary pixels, as an ``(P, 2)`` array.

    A boundary pixel is one inside the mask with at least one 4-neighbour
    outside it, which is the discrete surface the distance metrics measure
    against.
    """
    from scipy.ndimage import binary_erosion

    m = np.asarray(mask).astype(bool, copy=False)
    if not m.any():
        return np.zeros((0, m.ndim), dtype=np.int64)
    interior = binary_erosion(m, border_value=0)
    return np.argwhere(m & ~interior)


def hausdorff(a, b, *, percentile: float = 100.0) -> float:
    """Symmetric Hausdorff distance in pixels.

    ``percentile=95`` gives the HD95 variant, which is far less sensitive to a
    single stray component than the maximum the paper reports; both are recorded
    so the sensitivity of the conclusions to that choice can be measured.
    """
    if not 0 < percentile <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")

    pa, pb = surface_points(a), surface_points(b)
    if len(pa) == 0 and len(pb) == 0:
        return 0.0
    if len(pa) == 0 or len(pb) == 0:
        return float("inf")

    from scipy.spatial import cKDTree

    forward = cKDTree(pb).query(pa)[0]
    backward = cKDTree(pa).query(pb)[0]
    if percentile >= 100:
        return float(max(forward.max(), backward.max()))
    return float(
        max(np.percentile(forward, percentile), np.percentile(backward, percentile))
    )
