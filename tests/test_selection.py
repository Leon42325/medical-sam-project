"""Tests for mask selection and the oracle gap (PLAN.md Sec. 9.1).

Everything here runs on hand-built MaskSets, so the selection logic - the part
that decides what every reported DICE actually means - is verified without a GPU
or any model weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from samed.models.base import MaskSet, PromptableSegmenter, available, create, register
from samed.prompts import build_prompt
from samed.selection import SELECTION_RULES, oracle_gap, select


def _disc(shape=(64, 64), centre=(32, 32), radius=12) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.hypot(yy - centre[0], xx - centre[1]) <= radius


@pytest.fixture
def ground_truth() -> np.ndarray:
    return _disc()


@pytest.fixture
def masks(ground_truth) -> MaskSet:
    """Three candidates whose quality ordering disagrees with the model's scores.

    Index 1 is nearly perfect but the model rates it lowest; index 0 is the whole
    image and is rated highest. This is the situation the oracle rule hides.
    """
    whole_image = np.ones_like(ground_truth)
    good = _disc(radius=11)
    small = _disc(radius=4)
    return MaskSet(
        masks=np.stack([whole_image, good, small]),
        scores=np.array([0.9, 0.3, 0.5], dtype=np.float32),
        model="fake",
        strategy="S2",
    )


def test_oracle_rule_picks_the_best_mask_against_ground_truth(masks, ground_truth):
    chosen = select(masks, ground_truth, "oracle")
    assert chosen.index == 1
    assert chosen.dice > 0.9


def test_score_rule_can_pick_a_much_worse_mask(masks, ground_truth):
    chosen = select(masks, ground_truth, "score")
    assert chosen.index == 0, "the model's own head prefers the whole-image mask"
    assert chosen.dice < 0.5


def test_largest_and_first_rules(masks, ground_truth):
    assert select(masks, ground_truth, "largest").index == 0
    assert select(masks, ground_truth, "first").index == 0


def test_oracle_gap_is_the_difference_between_the_two_rules(masks, ground_truth):
    gap = oracle_gap(masks, ground_truth)
    expected = select(masks, ground_truth, "oracle").dice - select(masks, ground_truth, "score").dice
    assert gap == pytest.approx(expected)
    assert gap > 0.4, "this fixture is built to show a large gap"


@pytest.mark.parametrize("rule", [r for r in SELECTION_RULES if r != "oracle"])
def test_oracle_is_an_upper_bound_over_every_deployable_rule(masks, ground_truth, rule):
    """True by construction, and worth pinning: it is why the paper's numbers are bounds."""
    assert oracle_gap(masks, ground_truth, against=rule) >= 0


def test_oracle_gap_vanishes_when_the_quality_head_is_right(ground_truth):
    perfect = MaskSet(
        masks=np.stack([ground_truth, _disc(radius=3)]),
        scores=np.array([0.95, 0.10], dtype=np.float32),
        model="fake",
        strategy="S5",
    )
    assert oracle_gap(perfect, ground_truth) == pytest.approx(0.0)


def test_oracle_gap_refuses_to_compare_the_oracle_with_itself(masks, ground_truth):
    with pytest.raises(ValueError, match="deployable rule"):
        oracle_gap(masks, ground_truth, against="oracle")


def test_selection_rejects_a_shape_mismatch(masks):
    with pytest.raises(ValueError, match="does not match ground truth"):
        select(masks, np.zeros((32, 32), dtype=bool))


def test_selection_rejects_an_empty_candidate_set(ground_truth):
    empty = MaskSet(
        masks=np.zeros((0, 64, 64), dtype=bool),
        scores=np.zeros((0,), dtype=np.float32),
        model="fake",
        strategy="S1",
    )
    with pytest.raises(ValueError, match="no candidate masks"):
        select(empty, ground_truth)


def test_maskset_validates_its_own_shapes():
    with pytest.raises(ValueError, match=r"\(N, H, W\)"):
        MaskSet(np.zeros((8, 8), dtype=bool), np.zeros(1), "fake", "S5")
    with pytest.raises(ValueError, match="expected 3 scores"):
        MaskSet(np.zeros((3, 8, 8), dtype=bool), np.zeros(2), "fake", "S5")


# --------------------------------------------------------------------------- #
# The model interface, exercised through a stub so no weights are needed
# --------------------------------------------------------------------------- #


class _StubBoxOnly(PromptableSegmenter):
    name = "stub_box_only"
    supports = frozenset({"box"})
    input_size = 1024

    def encode(self, image):
        return {"features": np.zeros((1, 4, 4, 4), dtype=np.float16)}

    def predict(self, cached, prompt):
        return MaskSet(
            masks=np.ones((1, 8, 8), dtype=bool),
            scores=np.ones(1, dtype=np.float32),
            model=self.name,
            strategy=prompt.strategy,
        )


def test_box_only_models_reject_point_strategies():
    """MedSAM is box-only; S2-S4 must be reported N/A, never approximated."""
    stub = _StubBoxOnly()
    assert stub.supports_strategy("S5")
    assert stub.supports_strategy("S6") is False, "S6 needs a point as well as a box"
    for strategy in ("S1", "S2", "S3", "S4"):
        assert stub.supports_strategy(strategy) is False


def test_everything_mode_is_unavailable_unless_declared():
    with pytest.raises(NotImplementedError, match="everything mode"):
        _StubBoxOnly().everything(np.zeros((8, 8), dtype=np.uint8))


def test_registry_round_trip():
    @register("stub_for_registry_test")
    def _factory() -> PromptableSegmenter:
        return _StubBoxOnly()

    assert "stub_for_registry_test" in available()
    assert create("stub_for_registry_test").name == "stub_box_only"

    with pytest.raises(ValueError, match="already registered"):
        register("stub_for_registry_test")(_factory)
    with pytest.raises(KeyError, match="unknown model"):
        create("no_such_model")


def test_prompt_and_model_interfaces_line_up():
    """A prompt built for S5 must be consumable by a box-only model."""
    mask = _disc(shape=(8, 8), centre=(4, 4), radius=2)
    result = _StubBoxOnly().predict({}, build_prompt(mask, "S5"))
    assert result.strategy == "S5" and len(result) == 1
