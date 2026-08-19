"""Scoring candidate masks into result rows, shared by every prediction stage.

One schema for prompted and automatic modes, so that ``samed.analysis`` reads
them together without special cases and the selection rules mean the same thing
in both.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from samed.metrics import dice, hausdorff, jaccard

__all__ = ["FIELDS", "shard_filename", "score_candidates",
           "candidates_any_rule_would_pick"]


def shard_filename(stage: str, shard: int, num_shards: int, **variant) -> str:
    """Name a result shard so that no two runs can overwrite each other.

    Every dimension that changes what is inside the file has to appear in its
    name. Two have already caused silent damage: leaving out the shard *count*
    let a 16-way run reuse an 8-way run's output, and leaving out the *stage*
    let the automatic mode skip itself because the prompted stage had already
    written a file of that name into the same directory. Both looked like
    success - a completed job, no error, wrong or missing data.
    """
    parts = [stage, f"shard-{shard:04d}-of-{num_shards:04d}"]
    parts += [f"{key}{value}" for key, value in sorted(variant.items()) if value is not None]
    return "-".join(parts) + ".csv"


FIELDS = [
    "dataset", "modality", "target", "subject", "patient", "image_id", "slice_index",
    "label_value", "model", "strategy", "jitter", "jitter_points", "jitter_box_mode",
    "seed", "candidate", "n_candidates",
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
    jitter_points: str = "all",
    jitter_box_mode: str = "perturb",
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
            "model": model, "strategy": strategy, "jitter": jitter,
            "jitter_points": jitter_points, "jitter_box_mode": jitter_box_mode,
            "seed": seed,
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
