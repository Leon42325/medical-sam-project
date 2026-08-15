"""Deterministic subsampling protocol for the evaluation set.

Why this exists
---------------
COSMOS 1050K holds 1,050,311 images and 6,033,198 masks.  A course project on a
shared GPU cluster cannot evaluate six models under six prompting strategies on
that, so the study runs on a subset.  A subset chosen badly would silently
change the conclusions: the paper's own aggregation is dominated by its largest
datasets (TotalSegmentator alone contributes ~307k images), and naive slice
sampling would reproduce that bias one level down, letting a single large volume
speak for an entire anatomical target.

The protocol below is therefore fixed *before* any result exists, and is part of
the pre-registration:

1. Cap each object-modality target at ``max_per_target`` masks, so that targets
   contribute equally regardless of how large their source dataset is.
2. Within a target, spread the budget across subjects round-robin, so a patient
   with 400 annotated slices cannot outweigh forty patients with ten each.
3. Within a subject, take slices in bisecting order, so that any prefix of the
   budget covers the full extent of the structure - apex to base - rather than
   a contiguous run near one end.

Only step 2's tie-breaking uses randomness, and it is seeded; the selection is
otherwise fully determined by the inputs, so the evaluation set is reproducible
from the manifest alone.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = ["MaskRecord", "stratified_sample", "bisecting_order"]


@dataclass(frozen=True)
class MaskRecord:
    """One annotated object instance in one image.

    ``subject`` is the unit that must not be over-represented: a patient for
    clinical data, a case or acquisition otherwise.  ``slice_index`` orders
    slices within a 3D volume and is ``None`` for standalone 2D images.
    """

    dataset: str
    modality: str
    target: str
    subject: str
    image_id: str
    slice_index: int | None = None

    @property
    def target_key(self) -> tuple[str, str, str]:
        """The object-modality paired target, the unit the paper reports on."""
        return (self.dataset, self.modality, self.target)


def bisecting_order(n: int) -> list[int]:
    """Indices ``0..n-1`` ordered so that every prefix spans the whole range.

    Repeatedly bisects the remaining intervals breadth-first, giving
    ``[mid, mid of left half, mid of right half, ...]``.  Taking the first *k*
    entries yields *k* positions spread over the full extent, which is what
    makes a truncated per-subject budget still cover a structure end to end.
    """
    if n <= 0:
        return []
    order: list[int] = []
    queue: deque[tuple[int, int]] = deque([(0, n - 1)])
    while queue:
        low, high = queue.popleft()
        if low > high:
            continue
        mid = (low + high) // 2
        order.append(mid)
        queue.append((low, mid - 1))
        queue.append((mid + 1, high))
    return order


def _subject_order(records: Sequence[MaskRecord]) -> list[MaskRecord]:
    """Order one subject's records so that any prefix spans the structure."""
    ordered = sorted(
        records,
        key=lambda r: (r.slice_index if r.slice_index is not None else 0, r.image_id),
    )
    return [ordered[i] for i in bisecting_order(len(ordered))]


def stratified_sample(
    records: Iterable[MaskRecord],
    *,
    max_per_target: int,
    seed: int = 0,
) -> list[MaskRecord]:
    """Select at most ``max_per_target`` masks per object-modality target.

    Returns the selection in a stable order (target, then draw order), so two
    runs with the same inputs and seed produce byte-identical manifests.
    """
    if max_per_target <= 0:
        raise ValueError(f"max_per_target must be positive, got {max_per_target}")

    by_target: dict[tuple[str, str, str], list[MaskRecord]] = defaultdict(list)
    for record in records:
        by_target[record.target_key].append(record)

    selected: list[MaskRecord] = []
    for target_key in sorted(by_target):
        by_subject: dict[str, list[MaskRecord]] = defaultdict(list)
        for record in by_target[target_key]:
            by_subject[record.subject].append(record)

        # Seeded only to break the arbitrary ordering of equally eligible
        # subjects; which subjects appear at all is not left to chance.
        rng = np.random.default_rng([seed, *(ord(c) for c in "".join(target_key))])
        subjects = sorted(by_subject)
        rng.shuffle(subjects)

        queues = {s: deque(_subject_order(by_subject[s])) for s in subjects}
        budget = min(max_per_target, sum(len(q) for q in queues.values()))

        taken = 0
        while taken < budget:
            for subject in subjects:
                if taken >= budget:
                    break
                queue = queues[subject]
                if queue:
                    selected.append(queue.popleft())
                    taken += 1

    return selected
