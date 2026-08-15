"""Tests for the object attributes of Huang et al. (2024) Sec. 4.8."""

from __future__ import annotations

import numpy as np
import pytest

from samed.attributes import (
    area,
    aspect_ratio,
    fourier_order,
    intensity_difference,
    largest_contour,
)


def _polygon_mask(vertices: np.ndarray, size: int = 256) -> np.ndarray:
    import cv2

    canvas = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.round(vertices).astype(np.int32)], color=1)
    return canvas.astype(bool)


def _regular_polygon(n_vertices: int, radius: float, size: int = 256) -> np.ndarray:
    angle = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    centre = size / 2
    return _polygon_mask(
        np.stack([centre + radius * np.cos(angle), centre + radius * np.sin(angle)], axis=1),
        size,
    )


def _star(n_points: int, outer: float, inner: float, size: int = 256) -> np.ndarray:
    angle = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False)
    radius = np.where(np.arange(2 * n_points) % 2 == 0, outer, inner)
    centre = size / 2
    return _polygon_mask(
        np.stack([centre + radius * np.cos(angle), centre + radius * np.sin(angle)], axis=1),
        size,
    )


@pytest.fixture
def rectangle() -> np.ndarray:
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 20:60] = True  # 20 rows x 40 columns
    return mask


def test_area_counts_mask_pixels(rectangle):
    assert area(rectangle) == 20 * 40


def test_aspect_ratio_is_short_over_long(rectangle):
    assert aspect_ratio(rectangle) == pytest.approx(0.5)


def test_aspect_ratio_of_a_square_is_one():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    assert aspect_ratio(mask) == pytest.approx(1.0)


def test_intensity_difference_recovers_a_known_contrast(rectangle):
    image = np.full((64, 64), 100.0)
    image[rectangle] = 200.0
    assert intensity_difference(image, rectangle) == pytest.approx(100.0)


def test_intensity_difference_is_absolute_unless_signed_is_requested(rectangle):
    image = np.full((64, 64), 200.0)
    image[rectangle] = 100.0  # darker object on a brighter background
    assert intensity_difference(image, rectangle) == pytest.approx(100.0)
    assert intensity_difference(image, rectangle, signed=True) == pytest.approx(-100.0)


def test_intensity_difference_ignores_pixels_far_from_the_object():
    """Only the neighbourhood inside the enlarged box may influence the contrast."""
    image = np.full((128, 128), 100.0)
    mask = np.zeros((128, 128), dtype=bool)
    mask[60:68, 60:68] = True
    image[mask] = 200.0
    image[0:20, 0:20] = 0.0  # a bright/dark region far away must not matter

    assert intensity_difference(image, mask) == pytest.approx(100.0)


def test_intensity_difference_rejects_mismatched_shapes(rectangle):
    with pytest.raises(ValueError, match="does not match"):
        intensity_difference(np.zeros((10, 10)), rectangle)


def test_largest_contour_picks_the_bigger_component():
    mask = np.zeros((128, 128), dtype=bool)
    mask[10:20, 10:20] = True  # small
    mask[50:100, 50:100] = True  # large
    contour = largest_contour(mask)
    assert contour[:, 0].min() >= 50 and contour[:, 1].min() >= 50


def test_fourier_order_is_low_for_a_smooth_shape():
    """A near-circle is fitted by the first few harmonics."""
    result = fourier_order(_regular_polygon(64, radius=80))
    assert result.stopped_by == "dice_target"
    assert result.order <= 5
    assert result.value < 20


def test_fourier_order_grows_with_boundary_complexity():
    """The core premise of H1: FO must rank shapes by how convoluted they are."""
    circle = fourier_order(_regular_polygon(64, radius=80)).value
    blunt_star = fourier_order(_star(6, outer=80, inner=55)).value
    sharp_star = fourier_order(_star(14, outer=80, inner=30)).value

    assert circle < blunt_star < sharp_star


def test_fourier_order_is_scale_invariant_for_similar_shapes():
    """Size and complexity are separate factors in Table 6; FO must not track size."""
    small = fourier_order(_star(8, outer=40, inner=18)).value
    large = fourier_order(_star(8, outer=80, inner=36)).value
    assert small == pytest.approx(large, rel=0.35)


def test_fourier_order_penalty_charges_an_early_stop_at_a_poor_fit():
    """F_final = F_a + 2 * 100 * (1 - DICE); a poor fit must dominate the order."""
    result = fourier_order(_star(20, outer=100, inner=12), max_order=3)
    assert result.stopped_by in {"no_improvement", "max_order"}
    assert result.dice < 0.97
    assert result.value > result.order


@pytest.mark.parametrize(
    "n_points, expected_order",
    [(6, 11), (8, 13)],
)
def test_paper_criterion_two_stops_at_the_first_harmonic_plateau(n_points, expected_order):
    """The staircase flaw, pinned down as a regression test.

    A k-pointed star has almost no energy in harmonics 2..k-2, so the paper's
    "improvement < 0.1 %" rule terminates at order 2, while the contour is in
    fact only fitted many harmonics later.  Disabling the rule recovers the true
    order.
    """
    mask = _star(n_points, outer=80, inner=30)

    as_published = fourier_order(mask)
    assert as_published.stopped_by == "no_improvement"
    assert as_published.order == 2
    assert as_published.dice < 0.9

    corrected = fourier_order(mask, patience=None)
    assert corrected.stopped_by == "dice_target"
    assert corrected.order == expected_order


def test_patience_crosses_plateaus_shorter_than_itself():
    mask = _star(6, outer=80, inner=30)
    assert fourier_order(mask, patience=1).order == 2
    assert fourier_order(mask, patience=6).order > 2


def test_fourier_order_handles_a_degenerate_contour():
    mask = np.zeros((32, 32), dtype=bool)
    mask[10, 10] = True
    assert fourier_order(mask).stopped_by == "degenerate"


def test_attributes_reject_empty_masks():
    empty = np.zeros((16, 16), dtype=bool)
    with pytest.raises(ValueError):
        aspect_ratio(empty)
    with pytest.raises(ValueError):
        largest_contour(empty)
