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

S1 means something different
----------------------------
For the prompted strategies the prompt names the target, so picking the right
mask out of the candidates is genuinely the model's job and the gap measures a
deficiency in it.

Automatic mode names nothing. SAM segments whatever it finds and the outputs
carry no labels, so its quality head ranks masks by how cleanly they are
segmented, not by whether they are the kidney - and asking it to identify the
kidney is asking a question the mode does not answer. The S1 gap therefore is
not a ranking failure. It measures how much of the reported everything-mode
performance was supplied by the ground truth in the first place: an S1 oracle
score says a good mask exists somewhere in the output, not that any procedure
could find it.

The paper raises exactly this in its discussion - "how is semantics obtained
from SAM when there is no GT?" - and nonetheless reports S1 DICE as a
performance figure. The gap puts a number on the objection.
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
    "paired_comparison",
]

#: Columns that together identify one prompt, i.e. one set of candidate masks.
PROMPT_KEYS = [
    "dataset", "modality", "target", "subject", "patient", "image_id",
    "label_value", "model", "strategy", "jitter", "seed",
]


def load_results(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Read one or more shard CSVs, or every shard under a directory."""
    if isinstance(paths, (str, Path)):
        root = Path(paths)
        files = sorted(root.rglob("*shard-*.csv")) if root.is_dir() else [root]
    else:
        files = [Path(p) for p in paths]

    if not files:
        raise FileNotFoundError(f"no result shards found under {paths}")

    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    missing = set(PROMPT_KEYS + ["candidate", "predicted_iou", "dice"]) - set(frame.columns)
    if missing:
        raise ValueError(f"result files are missing columns: {sorted(missing)}")

    # Shards must partition the manifest. If they overlap - the usual cause is a
    # stale file from a run split a different number of ways - the duplicated
    # prompts are silently over-weighted in every mean that follows, so this is
    # an error rather than something to deduplicate quietly.
    duplicated = frame.duplicated(subset=PROMPT_KEYS + ["candidate"])
    if duplicated.any():
        example = frame.loc[duplicated, PROMPT_KEYS].iloc[0].to_dict()
        raise ValueError(
            f"{int(duplicated.sum())} duplicated candidate rows across "
            f"{len(files)} shard file(s); shards must not overlap. "
            f"First duplicate: {example}. "
            "Check for result files left over from a different --num-shards."
        )
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
    clusters: Sequence | None = None,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean.

    A bootstrap rather than a standard error because per-target DICE
    distributions are strongly skewed and often bimodal - a prompt either finds
    the organ or latches onto something else - so a symmetric interval around
    the mean would misdescribe the spread.

    ``clusters`` makes it a *cluster* bootstrap, resampling whole groups rather
    than individual observations. Pass the subject each measurement came from.

    This is not a refinement, it is the difference between a correct interval
    and a wrong one. Consecutive slices of one patient's scan are near-copies of
    each other, so 2409 masks from 40 patients carry nothing like 2409
    independent observations, and resampling masks individually reports an
    interval far narrower than the evidence supports. Criticising Huang et al.
    for treating 191,779 structures as i.i.d. (PLAN.md Sec. 9.2) while doing the
    same here would be indefensible.
    """
    data = np.asarray([v for v in values], dtype=float)
    finite = np.isfinite(data)
    if clusters is None:
        data = data[finite]
        if data.size == 0:
            return (float("nan"), float("nan"))
        if data.size == 1:
            return (float(data[0]), float(data[0]))

        rng = np.random.default_rng(seed)
        means = rng.choice(data, size=(resamples, data.size), replace=True).mean(axis=1)
    else:
        labels = np.asarray(list(clusters))[finite]
        data = data[finite]
        if data.size == 0:
            return (float("nan"), float("nan"))

        groups = [data[labels == label] for label in np.unique(labels)]
        if len(groups) == 1:
            return (float(data.mean()), float(data.mean()))

        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(groups), size=(resamples, len(groups)))
        means = np.array([
            np.concatenate([groups[i] for i in draw]).mean() for draw in indices
        ])

    tail = (1.0 - confidence) / 2.0
    return tuple(float(v) for v in np.percentile(means, [100 * tail, 100 * (1 - tail)]))


def summarise(
    selected: pd.DataFrame,
    *,
    by: Sequence[str] = ("modality", "target", "strategy"),
    cluster: str | None = "patient",
    seed: int = 0,
) -> pd.DataFrame:
    """Mean DICE under each rule, with cluster-bootstrap intervals.

    ``cluster`` names the column whose groups are resampled - the patient, by
    default. Not the series: CHAOS images each MR patient with two sequences,
    and treating those as two independent subjects overstates the evidence by
    half again. Set it
    to ``None`` only to reproduce the naive interval, which is narrower than the
    data justifies. ``n_clusters`` is reported alongside ``n`` so a reader can
    see how much independent evidence there actually is.
    """
    records = []
    for key, group in selected.groupby(list(by), sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        labels = group[cluster] if cluster and cluster in group else None
        low, high = bootstrap_ci(group["dice_score"], clusters=labels, seed=seed)
        gap_low, gap_high = bootstrap_ci(group["oracle_gap"], clusters=labels, seed=seed)
        records.append({
            **dict(zip(by, key)),
            "n": len(group),
            "n_clusters": int(labels.nunique()) if labels is not None else 0,
            "dice_oracle": group["dice_oracle"].mean(),
            "dice_score": group["dice_score"].mean(),
            "dice_score_lo": low,
            "dice_score_hi": high,
            "oracle_gap": group["oracle_gap"].mean(),
            "oracle_gap_lo": gap_low,
            "oracle_gap_hi": gap_high,
        })
    return pd.DataFrame.from_records(records)


def paired_comparison(
    selected: pd.DataFrame,
    *,
    baseline: str,
    other: str,
    by: Sequence[str] = ("strategy",),
    cluster: str | None = "patient",
    seed: int = 0,
) -> pd.DataFrame:
    """Difference between two models on the *same* prompts.

    Every model here sees an identical set of prompts, built deterministically
    from the same ground truth, so the comparison can be paired: the difference
    is taken prompt by prompt before it is averaged. That removes all the
    variance due to some organs simply being harder than others, which is the
    dominant source of spread in the unpaired tables and would otherwise swamp
    the effect being measured.

    Reported for both selection rules, because they can disagree about the sign.
    A model that produces better candidate masks while ranking them worse looks
    superior under the oracle rule and inferior in use - and the ranking is the
    half a user cannot fix.
    """
    keys = [k for k in PROMPT_KEYS if k != "model"]
    columns = ["dice_oracle", "dice_score"]

    left = selected[selected["model"] == baseline].set_index(keys)[columns]
    right = selected[selected["model"] == other].set_index(keys)[columns]
    if left.empty or right.empty:
        raise ValueError(
            f"need results for both {baseline!r} and {other!r}; "
            f"found {sorted(selected['model'].unique())}"
        )

    paired = right.join(left, how="inner", lsuffix="_other", rsuffix="_baseline")
    if paired.empty:
        raise ValueError("the two models share no prompts; were they run on one manifest?")
    paired = paired.reset_index()

    for column in columns:
        paired[f"delta_{column}"] = paired[f"{column}_other"] - paired[f"{column}_baseline"]

    records = []
    for key, group in paired.groupby(list(by), sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        labels = group[cluster] if cluster and cluster in group else None
        record = {**dict(zip(by, key)), "n_prompts": len(group)}
        for column in columns:
            delta = group[f"delta_{column}"]
            low, high = bootstrap_ci(delta, clusters=labels, seed=seed)
            record[f"delta_{column}"] = float(delta.mean())
            record[f"delta_{column}_lo"] = low
            record[f"delta_{column}_hi"] = high
            # An interval clear of zero on both sides is a difference the data
            # supports; the sign is what the two rules can disagree about.
            record[f"delta_{column}_sig"] = "yes" if low * high > 0 else "no"
        records.append(record)
    return pd.DataFrame.from_records(records)
