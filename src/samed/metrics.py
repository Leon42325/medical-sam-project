"""Segmentation metrics.

Currently holds only the overlap metric needed by :mod:`samed.attributes`.
The full suite used for evaluation (JAC, HD, and the boundary-aware NSD and
Boundary-IoU that we add as an evaluation improvement) lands in
``samed.eval`` and will reuse :func:`dice` from here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["dice"]


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
