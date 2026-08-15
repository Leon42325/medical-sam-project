"""Turning per-candidate predictions into the numbers a reader can act on.

``samed.cli.predict`` writes one row per candidate mask. This module collapses
those to one row per *prompt* under each selection rule, and aggregates.

The central quantity is the difference between two ways of choosing among the
candidates SAM returns:

* the **oracle** rule, which keeps whichever mask scores best against the ground
  truth. This is what Huang et al. use (Sec. 3.5), and it is not available to
  anyone segmenting an unlabelled image.
* a **deployable** rule, which keeps whichever mask the model's own quality head
  rates highest.

Their difference is the amount by which the published numbers exceed what the
method can actually deliver. It is not uniform: it shrinks as a prompt becomes
less ambiguous, which means the oracle rule does not merely inflate scores, it
compresses the differences *between* prompting strategies - the very effects the
paper's conclusions rest on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PROMPT_KEYS",
    "load_results",
    "select_per_prompt",
    "summarise",
    "bootstrap_ci",
]

#: Columns that together identify one prompt, i.e. one set of candidate masks.
PROMPT_KEYS = [
    "dataset", "modality", "target", "subject", "image_id",
    "label_value", "model", "strategy", "jitter", "seed",
]


def load_results(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Read one or more shard CSVs, or every shard under a directory."""
    if isinstance(paths, (str, Path)):
        root = Path(paths)
        files = sorted(root.rglob("shard-*.csv")) if root.is_dir() else [root]
    else:
        files = [Path(p) for p in paths]

    if not files:
        raise FileNotFoundError(f"no result shards found under {paths}")

    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    missing = set(PROMPT_KEYS + ["candidate", "predicted_iou", "dice"]) - set(frame.columns)
    if missing:
        raise ValueError(f"result files are missing columns: {sorted(missing)}")
    return frame


def select_per_prompt(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse candidates to one row per prompt, under each selection rule.

    Adds ``dice_oracle`` (the paper's rule), ``dice_score`` (the model's own
    quality head), ``dice_first`` (naive baseline) and ``oracle_gap``. Distance
    metrics are carried along for whichever candidate each rule chose, so that
    HD is reported for the mask a rule would actually return rather than for a
    different one.
    """
    grouped = results.groupby(PROMPT_KEYS, sort=False, dropna=False)

    oracle_index = grouped["dice"].idxmax()
    score_index = grouped["predicted_iou"].idxmax()
    first_index = grouped["candidate"].idxmin()

    def rows_at(index: pd.Series, suffix: str) -> pd.DataFrame:
        chosen = results.loc[index, PROMPT_KEYS + ["dice", "jaccard", "hd", "hd95", "candidate"]]
        return chosen.rename(columns={
            column: f"{column}_{suffix}"
            for column in ("dice", "jaccard", "hd", "hd95", "candidate")
        }).set_index(PROMPT_KEYS)

    selected = rows_at(oracle_index, "oracle").join(
        [rows_at(score_index, "score"), rows_at(first_index, "first")]
    ).reset_index()
    selected["oracle_gap"] = selected["dice_oracle"] - selected["dice_score"]
    return selected


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean.

    Reported instead of a standard error because per-target DICE distributions
    are strongly skewed and often bimodal - a prompt either finds the organ or
    latches onto something else - so a symmetric interval around the mean would
    misdescribe the spread.
    """
    data = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if data.size == 0:
        return (float("nan"), float("nan"))
    if data.size == 1:
        return (float(data[0]), float(data[0]))

    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(resamples, data.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return tuple(float(v) for v in np.percentile(means, [100 * tail, 100 * (1 - tail)]))


def summarise(
    selected: pd.DataFrame,
    *,
    by: Sequence[str] = ("modality", "target", "strategy"),
    seed: int = 0,
) -> pd.DataFrame:
    """Mean DICE under each rule, with bootstrap intervals, grouped by ``by``."""
    records = []
    for key, group in selected.groupby(list(by), sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        low, high = bootstrap_ci(group["dice_score"], seed=seed)
        gap_low, gap_high = bootstrap_ci(group["oracle_gap"], seed=seed)
        records.append({
            **dict(zip(by, key)),
            "n": len(group),
            "dice_oracle": group["dice_oracle"].mean(),
            "dice_score": group["dice_score"].mean(),
            "dice_score_lo": low,
            "dice_score_hi": high,
            "oracle_gap": group["oracle_gap"].mean(),
            "oracle_gap_lo": gap_low,
            "oracle_gap_hi": gap_high,
        })
    return pd.DataFrame.from_records(records)
