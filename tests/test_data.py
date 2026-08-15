"""Tests for the preprocessing protocol (paper Sec. 2.2) and the sampling protocol."""

from __future__ import annotations

import numpy as np
import pytest

from samed.data.preprocess import (
    min_max_normalise,
    select_labelled_slices,
    slice_label_areas,
)
from samed.data.sampling import MaskRecord, bisecting_order, stratified_sample


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_min_max_normalise_maps_the_range_onto_0_255():
    volume = np.array([[-2000.0, 0.0], [1000.0, 2000.0]])
    out = min_max_normalise(volume)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_min_max_normalise_is_volume_wide_by_default():
    """Slice-wise scaling would erase the intensity relationship between slices."""
    volume = np.stack([np.full((4, 4), 10.0), np.full((4, 4), 200.0)])
    volume[0, 0, 0] = 0.0
    volume[1, 0, 0] = 255.0

    whole = min_max_normalise(volume, scope="volume")
    assert whole[0].max() < whole[1].max(), "a darker slice must stay darker"

    per_slice = min_max_normalise(volume, scope="slice", axis=0)
    assert per_slice[0].max() == per_slice[1].max() == 255


def test_min_max_normalise_survives_a_constant_image():
    out = min_max_normalise(np.full((8, 8), 42.0))
    assert out.dtype == np.uint8 and np.all(out == 0)


def test_slice_label_areas_counts_pixels_not_category_codes():
    """A single pixel encoded as 85 must not pass a threshold meant to mean 50 pixels."""
    labels = np.zeros((3, 10, 10), dtype=np.uint8)
    labels[0, 0, 0] = 85
    labels[1, :8, :8] = 170
    areas = slice_label_areas(labels, axis=0)
    assert areas.tolist() == [1, 64, 0]


def test_slice_label_areas_can_isolate_one_category():
    labels = np.zeros((2, 10, 10), dtype=np.uint8)
    labels[0, :6, :6] = 85
    labels[0, 6:9, 6:9] = 170
    assert slice_label_areas(labels, axis=0, label_value=85).tolist() == [36, 0]


def test_select_labelled_slices_applies_a_strict_threshold():
    labels = np.zeros((4, 20, 20), dtype=np.uint8)
    labels[0, :7, :7] = 1  # 49 pixels -> dropped
    labels[1, :50] = 0
    labels[2, :10, :5] = 1  # 50 pixels -> dropped, threshold is "greater than"
    labels[3, :10, :6] = 1  # 60 pixels -> kept
    assert select_labelled_slices(labels, axis=0).tolist() == [3]


def test_select_labelled_slices_honours_a_custom_threshold():
    labels = np.zeros((2, 10, 10), dtype=np.uint8)
    labels[0, :3, :3] = 1  # 9 pixels
    assert select_labelled_slices(labels, axis=0, min_area=5).tolist() == [0]


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def test_bisecting_order_is_a_permutation():
    for n in (0, 1, 2, 5, 16, 33):
        assert sorted(bisecting_order(n)) == list(range(n))


@pytest.mark.parametrize("prefix", [4, 8, 16])
def test_bisecting_order_prefixes_span_the_range(prefix):
    order = bisecting_order(100)[:prefix]
    assert min(order) < 40 and max(order) > 60


def test_bisecting_order_never_returns_a_contiguous_run():
    """The failure mode that matters: a budget spent on one end of the structure.

    Two points can only ever be a midpoint plus a quarter-point, so the guarantee
    for very short prefixes is separation rather than full coverage.
    """
    for prefix in (2, 3, 4, 8):
        order = sorted(bisecting_order(100)[:prefix])
        assert max(order) - min(order) >= 20
        assert order != list(range(order[0], order[0] + prefix))


def _volume(subject: str, n: int, target: str = "liver") -> list[MaskRecord]:
    return [
        MaskRecord("ds", "CT", target, subject, f"{subject}_{i:04d}", slice_index=i)
        for i in range(n)
    ]


def test_sample_respects_the_per_target_cap():
    records = _volume("a", 500) + _volume("b", 500)
    assert len(stratified_sample(records, max_per_target=100)) == 100


def test_sample_returns_everything_when_the_budget_exceeds_the_data():
    records = _volume("a", 7)
    assert len(stratified_sample(records, max_per_target=100)) == 7


def test_sample_caps_each_target_independently():
    records = _volume("a", 300) + _volume("b", 300, target="spleen")
    picked = stratified_sample(records, max_per_target=50)
    assert len(picked) == 100
    assert sum(r.target == "liver" for r in picked) == 50
    assert sum(r.target == "spleen" for r in picked) == 50


def test_one_huge_volume_cannot_dominate_a_target():
    """The bias this protocol exists to prevent."""
    records = _volume("big", 1000)
    for i in range(9):
        records += _volume(f"small{i}", 10)

    picked = stratified_sample(records, max_per_target=100)
    from_big = sum(r.subject == "big" for r in picked)

    assert from_big <= 20, f"the 1000-slice volume took {from_big} of 100 slots"
    assert len({r.subject for r in picked}) == 10, "every subject must be represented"


def test_small_subjects_are_fully_used_before_big_ones_are_over_drawn():
    records = _volume("big", 100) + _volume("small", 3)
    picked = stratified_sample(records, max_per_target=20)
    assert sum(r.subject == "small" for r in picked) == 3


def test_selected_slices_span_the_structure():
    """A contiguous run near one end would bias every attribute we measure."""
    records = _volume("a", 200)
    indices = [r.slice_index for r in stratified_sample(records, max_per_target=10)]
    assert min(indices) < 40 and max(indices) > 160


def test_sample_is_deterministic_for_a_given_seed():
    records = _volume("a", 100) + _volume("b", 100) + _volume("c", 100)
    first = stratified_sample(records, max_per_target=40, seed=7)
    second = stratified_sample(records, max_per_target=40, seed=7)
    assert first == second


def test_sample_size_is_stable_across_seeds():
    records = _volume("a", 100) + _volume("b", 100)
    sizes = {len(stratified_sample(records, max_per_target=30, seed=s)) for s in range(5)}
    assert sizes == {30}


def test_sample_rejects_a_non_positive_budget():
    with pytest.raises(ValueError, match="must be positive"):
        stratified_sample(_volume("a", 5), max_per_target=0)


def test_records_expose_the_object_modality_target_key():
    record = MaskRecord("CHAOS", "T1W-MRI", "liver", "p01", "img0", slice_index=3)
    assert record.target_key == ("CHAOS", "T1W-MRI", "liver")
