"""Partitioning the evaluation set for fine-tuning, by patient.

Fine-tuning turns a benchmark into an experiment, and an experiment needs a
split. The requirement that matters is that it is a split *by patient*: adjacent
slices of one scan are near-copies, so putting some of a patient's slices in
training and the rest in test measures memorisation and reports it as
generalisation. Nothing downstream would flag it - the numbers simply come out
better.

Splitting by patient is necessary but not sufficient. The comparison this
supports is per object-modality target, so every target must appear on both
sides; a split that happened to put all the T2W spleen patients in training
would silently drop that target from the results. Patients are therefore
partitioned *within* each (dataset, modality) stratum, and the coverage is
checked rather than assumed.

The zero-shot results already computed cover every patient. They do not need
recomputing: filtering them to the test patients gives the matched baseline,
since the prompts are built deterministically from the ground truth and do not
depend on which split a patient landed in.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from samed.data.manifest import ManifestRow

__all__ = ["DEFAULT_FRACTIONS", "split_by_patient", "coverage_gaps", "describe_split"]

#: Enough test patients to say something, enough training patients to learn from.
DEFAULT_FRACTIONS: Mapping[str, float] = {"train": 0.6, "val": 0.15, "test": 0.25}


def _assign(patients: Sequence[str], fractions: Mapping[str, float],
            rng: np.random.Generator) -> dict[str, list[str]]:
    """Deal patients into splits, giving every split at least one."""
    names = list(fractions)
    if len(patients) < len(names):
        raise ValueError(
            f"cannot fill {len(names)} splits from {len(patients)} patient(s): "
            f"{sorted(patients)}. Use fewer splits or a larger stratum."
        )

    order = list(patients)
    rng.shuffle(order)

    # Largest-remainder allocation, then a guaranteed minimum of one each, so a
    # small stratum cannot leave a split empty and drop its targets entirely.
    exact = {name: fractions[name] * len(order) for name in names}
    counts = {name: max(1, int(exact[name])) for name in names}
    while sum(counts.values()) > len(order):
        counts[max(counts, key=lambda n: counts[n] - exact[n])] -= 1
    while sum(counts.values()) < len(order):
        counts[max(counts, key=lambda n: exact[n] - counts[n])] += 1

    splits, start = {}, 0
    for name in names:
        splits[name] = order[start:start + counts[name]]
        start += counts[name]
    return splits


def split_by_patient(
    rows: Iterable[ManifestRow],
    *,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
    stratify_by: Sequence[str] = ("dataset", "modality"),
    seed: int = 0,
) -> dict[str, list[ManifestRow]]:
    """Partition manifest rows so that no patient appears in two splits."""
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {dict(fractions)}")

    rows = list(rows)
    strata: dict[tuple, set[str]] = defaultdict(set)
    for row in rows:
        strata[tuple(getattr(row, key) for key in stratify_by)].add(row.patient or row.subject)

    owner: dict[str, str] = {}
    for stratum in sorted(strata):
        rng = np.random.default_rng([seed, *(ord(c) for c in "".join(map(str, stratum)))])
        for name, patients in _assign(sorted(strata[stratum]), fractions, rng).items():
            for patient in patients:
                owner[patient] = name

    splits: dict[str, list[ManifestRow]] = {name: [] for name in fractions}
    for row in rows:
        splits[owner[row.patient or row.subject]].append(row)
    return splits


def coverage_gaps(splits: Mapping[str, Sequence[ManifestRow]]) -> dict[str, list[tuple]]:
    """Object-modality targets missing from each split.

    An empty result is the precondition for reporting per-target improvements:
    a target absent from training cannot be learned, and one absent from test
    cannot be measured.
    """
    def targets(rows: Sequence[ManifestRow]) -> set[tuple]:
        return {(r.dataset, r.modality, r.target) for r in rows}

    everything = set().union(*(targets(rows) for rows in splits.values())) if splits else set()
    return {
        name: sorted(everything - targets(rows))
        for name, rows in splits.items()
        if everything - targets(rows)
    }


def describe_split(splits: Mapping[str, Sequence[ManifestRow]]) -> str:
    lines = [f"{'split':<8}{'rows':>8}{'patients':>10}{'targets':>10}"]
    for name, rows in splits.items():
        patients = {r.patient or r.subject for r in rows}
        targets = {(r.modality, r.target) for r in rows}
        lines.append(f"{name:<8}{len(rows):>8}{len(patients):>10}{len(targets):>10}")
    return "\n".join(lines)
