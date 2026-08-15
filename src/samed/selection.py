"""Choosing one mask out of the several a promptable model returns.

This module exists because of a flaw in the original evaluation.  Huang et al.
select, among the N masks SAM emits for a prompt, the one with the highest DICE
*against the ground truth* (Sec. 3.5, Eq. 1):

    "we calculated a set of dice scores between N binary predicted masks and the
    GT G.  Then, the one with the highest dice score in the set was selected as
    the matched predicted mask P for subsequent segmentation evaluation."

That rule consults the ground truth at inference time.  No deployment can do
this - if the ground truth were available, there would be nothing to segment -
so every number in the paper is an *oracle* upper bound rather than an
attainable score.  The gap is expected to be largest exactly where N is largest,
i.e. under the single-point strategies where SAM's output is most ambiguous.

Since all candidate masks are retained (see :class:`samed.models.MaskSet`),
re-scoring under a deployable rule costs no extra inference.  Both rules are
therefore computed for every prediction and reported side by side, and their
difference - the oracle gap - is itself a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from samed.metrics import dice
from samed.models.base import MaskSet

__all__ = ["Selection", "select", "oracle_gap", "SELECTION_RULES"]

Rule = Literal["oracle", "score", "largest", "first"]

SELECTION_RULES: tuple[Rule, ...] = ("oracle", "score", "largest", "first")


@dataclass(frozen=True)
class Selection:
    """The mask a rule picked, and how good it turned out to be."""

    mask: np.ndarray
    index: int
    dice: float
    rule: Rule


def select(masks: MaskSet, ground_truth, rule: Rule = "score") -> Selection:
    """Apply one selection rule and score the result against the ground truth.

    ``ground_truth`` is always required - not to make the choice (except under
    ``"oracle"``), but to report the DICE of whatever was chosen.

    Rules:

    * ``oracle`` - highest DICE against the ground truth.  The paper's rule.
      Reported as an upper bound, never as a deployable score.
    * ``score``  - highest predicted IoU from the model's own quality head.  What
      an actual user of the API gets, and the default here.
    * ``largest`` - largest mask by area; a common heuristic when no quality head
      is available.
    * ``first`` - the model's first output, the naive baseline.
    """
    if len(masks) == 0:
        raise ValueError("no candidate masks to select from")

    gt = np.asarray(ground_truth).astype(bool, copy=False)
    if masks.masks.shape[1:] != gt.shape:
        raise ValueError(
            f"mask shape {masks.masks.shape[1:]} does not match ground truth {gt.shape}"
        )

    if rule == "oracle":
        scores = np.array([dice(m, gt) for m in masks.masks])
        index = int(np.argmax(scores))
    elif rule == "score":
        index = int(np.argmax(masks.scores))
    elif rule == "largest":
        index = int(np.argmax(masks.masks.reshape(len(masks), -1).sum(axis=1)))
    elif rule == "first":
        index = 0
    else:  # pragma: no cover - guarded by typing
        raise ValueError(f"unknown selection rule {rule!r}")

    chosen = masks.masks[index]
    return Selection(mask=chosen, index=index, dice=dice(chosen, gt), rule=rule)


def oracle_gap(masks: MaskSet, ground_truth, *, against: Rule = "score") -> float:
    """DICE lost by using a deployable rule instead of the paper's oracle.

    Non-negative by construction: the oracle maximises DICE over the same
    candidate set. A gap of zero means the model's own quality head already
    identifies the best mask, and the paper's number is attainable after all.
    """
    if against == "oracle":
        raise ValueError("the oracle gap is measured against a deployable rule")
    return select(masks, ground_truth, "oracle").dice - select(masks, ground_truth, against).dice
