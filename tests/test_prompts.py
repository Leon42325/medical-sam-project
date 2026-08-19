"""Tests for the S1-S6 prompt construction rules of Huang et al. (2024) Sec. 3.3."""

from __future__ import annotations

import numpy as np
import pytest

from samed.prompts import (
    STRATEGY_SPEC,
    bounding_box,
    build_prompt,
    center_of_mass,
    jitter_prompt,
    negative_points,
    positive_points,
)


@pytest.fixture
def square() -> np.ndarray:
    """A 20x20 filled square inside a 64x64 image; centre of mass is inside."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 25:45] = True
    return mask


@pytest.fixture
def two_blobs() -> np.ndarray:
    """Two separated blobs; the centre of mass falls in the gap between them."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[28:36, 6:16] = True
    mask[28:36, 48:58] = True
    return mask


@pytest.fixture
def annulus() -> np.ndarray:
    """A ring; the centre of mass falls in the hole, i.e. outside the mask."""
    yy, xx = np.mgrid[0:64, 0:64]
    radius = np.hypot(yy - 32, xx - 32)
    return (radius > 12) & (radius < 22)


def test_bounding_box_is_tight_and_inclusive(square):
    assert bounding_box(square).tolist() == [25.0, 20.0, 44.0, 39.0]


def test_bounding_box_rejects_empty_mask():
    with pytest.raises(ValueError, match="empty"):
        bounding_box(np.zeros((8, 8), dtype=bool))


def test_centre_of_mass_becomes_the_first_positive_point(square):
    cx, cy = center_of_mass(square)
    points = positive_points(square, 5)
    assert points[0].tolist() == [round(cx), round(cy)]


def test_centre_of_mass_is_skipped_when_it_falls_outside(two_blobs, annulus):
    for mask in (two_blobs, annulus):
        cx, cy = center_of_mass(mask)
        assert not mask[int(round(cy)), int(round(cx))], "fixture must have its CoM outside"
        points = positive_points(mask, 5)
        assert len(points) == 5
        assert not np.allclose(points[0], [round(cx), round(cy)])


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_positive_points_always_land_inside_the_mask(square, two_blobs, annulus, n):
    for mask in (square, two_blobs, annulus):
        for x, y in positive_points(mask, n):
            assert mask[int(y), int(x)], f"positive point ({x}, {y}) is outside the mask"


def test_positive_points_are_spread_over_all_components(two_blobs):
    """Uniform sampling of the flattened mask must not collapse onto one blob."""
    xs = positive_points(two_blobs, 5)[:, 0]
    assert (xs < 32).any() and (xs > 32).any()


@pytest.mark.parametrize("n", [1, 5, 10])
def test_negative_points_are_outside_the_mask_but_inside_the_enlarged_box(square, n):
    x0, y0, x1, y1 = bounding_box(square)
    # enlarge=2.0 -> the half-extent of the window equals one full side length.
    half_w, half_h = (x1 - x0 + 1), (y1 - y0 + 1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # The window is snapped outwards to whole pixels before sampling.
    left, right = np.floor(cx - half_w), np.ceil(cx + half_w)
    top, bottom = np.floor(cy - half_h), np.ceil(cy + half_h)

    for x, y in negative_points(square, n):
        assert not square[int(y), int(x)], f"negative point ({x}, {y}) is inside the mask"
        assert left <= x <= right
        assert top <= y <= bottom


def test_negative_points_fail_loudly_when_the_object_fills_its_neighbourhood():
    with pytest.raises(ValueError, match="no background pixels"):
        negative_points(np.ones((16, 16), dtype=bool), 5)


def test_s1_carries_no_prompt(square):
    prompt = build_prompt(square, "S1")
    assert prompt.point_coords is None and prompt.point_labels is None and prompt.box is None


@pytest.mark.parametrize("strategy", ["S2", "S3", "S4", "S5", "S6"])
def test_strategy_shapes_match_the_specification(square, strategy):
    spec = STRATEGY_SPEC[strategy]
    prompt = build_prompt(square, strategy)

    assert prompt.n_points == spec["n_pos"] + spec["n_neg"]
    assert (prompt.box is not None) == spec["box"]
    if prompt.point_labels is not None:
        assert int((prompt.point_labels == 1).sum()) == spec["n_pos"]
        assert int((prompt.point_labels == 0).sum()) == spec["n_neg"]


def test_s4_orders_positive_points_before_negative_ones(square):
    labels = build_prompt(square, "S4").point_labels
    assert labels.tolist() == [1] * 5 + [0] * 5


def test_construction_is_deterministic(annulus):
    """The paper builds prompts to be repeatable; identical input, identical output."""
    for strategy in ("S2", "S3", "S4", "S5", "S6"):
        first = build_prompt(annulus, strategy)
        second = build_prompt(annulus, strategy)
        np.testing.assert_array_equal(first.point_coords, second.point_coords)
        np.testing.assert_array_equal(first.box, second.box)


def test_sampling_scheme_changes_which_points_are_picked(square):
    centers = positive_points(square, 5, scheme="centers")
    endpoints = positive_points(square, 5, scheme="endpoints")
    assert not np.array_equal(centers, endpoints)


def test_jitter_displaces_points_within_the_requested_band(square):
    rng = np.random.default_rng(0)
    prompt = build_prompt(square, "S4")
    jittered = jitter_prompt(prompt, 10, 20, rng)

    distance = np.linalg.norm(jittered.point_coords - prompt.point_coords, axis=1)
    assert np.all(distance >= 10 - 1e-6) and np.all(distance <= 20 + 1e-6)
    np.testing.assert_array_equal(jittered.point_labels, prompt.point_labels)


def test_jitter_shift_mode_translates_the_box_rigidly(square):
    rng = np.random.default_rng(1)
    prompt = build_prompt(square, "S5")
    jittered = jitter_prompt(prompt, 5, 10, rng, box_mode="shift")

    original_size = prompt.box[2:] - prompt.box[:2]
    jittered_size = jittered.box[2:] - jittered.box[:2]
    np.testing.assert_allclose(original_size, jittered_size, atol=1e-5)


def test_jitter_perturb_mode_also_resizes_the_box(square):
    rng = np.random.default_rng(2)
    prompt = build_prompt(square, "S5")
    jittered = jitter_prompt(prompt, 5, 10, rng, box_mode="perturb")

    original_size = prompt.box[2:] - prompt.box[:2]
    jittered_size = jittered.box[2:] - jittered.box[:2]
    assert not np.allclose(original_size, jittered_size)


def test_jitter_clipping_keeps_the_box_inside_the_image_and_well_formed(square):
    rng = np.random.default_rng(3)
    prompt = build_prompt(square, "S6")
    jittered = jitter_prompt(prompt, 25, 30, rng, image_shape=square.shape)

    x0, y0, x1, y1 = jittered.box
    assert 0 <= x0 <= x1 <= square.shape[1] - 1
    assert 0 <= y0 <= y1 <= square.shape[0] - 1
    assert np.all(jittered.point_coords[:, 0] <= square.shape[1] - 1)
    assert np.all(jittered.point_coords[:, 1] <= square.shape[0] - 1)


def test_unknown_strategy_is_rejected(square):
    with pytest.raises(ValueError, match="unknown strategy"):
        build_prompt(square, "S7")


def test_jitter_can_move_only_the_centre_point(square):
    """Which points move decides one of the paper's conclusions.

    The paper adds randomness "to the centers and boxes" and finds that more
    points make SAM more robust. That follows only if the extra points stay
    where they were: displacing all five leaves a five-point prompt no steadier
    than a single one.
    """
    rng = np.random.default_rng(0)
    prompt = build_prompt(square, "S3")

    every = jitter_prompt(prompt, 20, 30, rng, points="all")
    moved = np.linalg.norm(every.point_coords - prompt.point_coords, axis=1)
    assert (moved > 1).all(), "all five points move"

    rng = np.random.default_rng(0)
    centre_only = jitter_prompt(prompt, 20, 30, rng, points="first")
    moved = np.linalg.norm(centre_only.point_coords - prompt.point_coords, axis=1)
    assert moved[0] >= 20 - 1e-6
    assert (moved[1:] == 0).all(), "the uniformly sampled points stay put"


def test_centre_only_jitter_leaves_negative_points_alone(square):
    rng = np.random.default_rng(1)
    prompt = build_prompt(square, "S4")
    jittered = jitter_prompt(prompt, 20, 30, rng, points="first")

    negatives = prompt.point_labels == 0
    np.testing.assert_array_equal(
        jittered.point_coords[negatives], prompt.point_coords[negatives]
    )


def test_the_two_point_readings_differ_measurably(square):
    rng = np.random.default_rng(2)
    prompt = build_prompt(square, "S4")
    every = jitter_prompt(prompt, 20, 30, np.random.default_rng(2), points="all")
    centre = jitter_prompt(prompt, 20, 30, np.random.default_rng(2), points="first")

    total_all = np.linalg.norm(every.point_coords - prompt.point_coords, axis=1).sum()
    total_one = np.linalg.norm(centre.point_coords - prompt.point_coords, axis=1).sum()
    assert total_all > 5 * total_one * 0.8
