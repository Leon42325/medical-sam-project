"""Scoring candidate masks into result rows, shared by every prediction stage.

One schema for prompted and automatic modes, so that ``samed.analysis`` reads
them together without special cases and the selection rules mean the same thing
in both.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from samed.metrics import dice, hausdorff, jaccard

__all__ = ["FIELDS", "score_candidates", "candidates_any_rule_would_pick"]

FIELDS = [
    "dataset", "modality", "target", "subject", "patient", "image_id", "slice_index",
    "label_value", "model", "strategy", "jitter", "seed", "candidate", "n_candidates",
    "predicted_iou", "dice", "jaccard", "hd", "hd95", "gt_area", "pred_area",
]


def candidates_any_rule_would_pick(masks, ground_truth) -> list[int]:
    """Indices worth keeping when a model returns many masks.

    Prompted modes emit about three candidates and all of them are stored. The
    automatic mode emits tens to hundreds per image, and a Hausdorff distance
    for each against every target would dominate the run time while adding
    nothing: the analysis only ever asks which mask a *rule* would return.

    Keeping the mask chosen by each rule - best against the ground truth, best
    by the model's own quality head, largest, and first - reproduces every rule
    exactly, at bounded cost. ``n_candidates`` records how many there were, so
    the reduction stays visible in the results.
    """
    if len(masks) == 0:
        return []

    overlaps = np.array([dice(mask, ground_truth) for mask in masks.masks])
    areas = masks.masks.reshape(len(masks), -1).sum(axis=1)
    return sorted({
        int(np.argmax(overlaps)),       # the paper's oracle rule
        int(np.argmax(masks.scores)),   # the model's own quality head
        int(np.argmax(areas)),          # a common heuristic without one
        0,                              # naive baseline
    })


def score_candidates(
    masks,
    ground_truth,
    row,
    *,
    model: str,
    strategy: str,
    jitter: str = "none",
    seed: int = 0,
    indices: Sequence[int] | None = None,
) -> list[dict]:
    """One result row per candidate mask, or per entry of ``indices``."""
    gt_area = int(ground_truth.sum())
    chosen = range(len(masks)) if indices is None else indices

    rows = []
    for index in chosen:
        candidate = masks.masks[index]
        rows.append({
            "dataset": row.dataset, "modality": row.modality, "target": row.target,
            "subject": row.subject, "patient": row.patient, "image_id": row.image_id,
            "slice_index": row.slice_index, "label_value": row.label_value,
            "model": model, "strategy": strategy, "jitter": jitter, "seed": seed,
            "candidate": int(index), "n_candidates": len(masks),
            "predicted_iou": float(masks.scores[index]),
            "dice": dice(candidate, ground_truth),
            "jaccard": jaccard(candidate, ground_truth),
            "hd": hausdorff(candidate, ground_truth),
            "hd95": hausdorff(candidate, ground_truth, percentile=95),
            "gt_area": gt_area,
            "pred_area": int(candidate.sum()),
        })
    return rows
