"""Tests for checkpoint verification.

The comparison logic is exercised with small tensors, so the checks that decide
whether a downloaded file may be called "MedSAM" are themselves tested without
needing a 2.4 GB download or a GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="checkpoint checks operate on torch tensors")

from samed.cli.checkpoint import compare, describe, summarise_sections  # noqa: E402


def _state(scale: float = 1.0, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "image_encoder.blocks.0.weight": torch.randn(4, 4, generator=generator) * scale,
        "image_encoder.blocks.1.weight": torch.randn(4, 4, generator=generator) * scale,
        "prompt_encoder.embed.weight": torch.randn(2, 2, generator=generator) * scale,
        "mask_decoder.head.weight": torch.randn(3, 3, generator=generator) * scale,
    }


def test_describe_counts_parameters_by_section():
    facts = describe(_state())
    assert facts["tensors"] == 4
    assert facts["parameters"] == 4 * 4 + 4 * 4 + 2 * 2 + 3 * 3
    assert facts["sections"]["image_encoder"] == 32
    assert facts["sections"]["mask_decoder"] == 9


def test_an_unchanged_copy_is_reported_as_identical():
    """The failure mode: vanilla SAM re-uploaded under a medical name."""
    reference = _state()
    result = compare(dict(reference), reference)
    assert result["shared"] == 4
    assert result["identical"] == 4, "every tensor is bit-identical"
    assert all(value == 0.0 for value in result["sections"].values())


def test_a_decoder_only_finetune_shows_where_it_moved():
    """MedSAM froze the image encoder, so divergence must concentrate downstream."""
    reference = _state()
    candidate = {key: tensor.clone() for key, tensor in reference.items()}
    candidate["mask_decoder.head.weight"] += 1.0

    result = compare(candidate, reference)
    assert result["identical"] == 3
    assert result["sections"]["image_encoder"] == 0.0
    assert result["sections"]["mask_decoder"] == pytest.approx(1.0)


def test_comparison_skips_tensors_whose_shape_changed():
    reference = _state()
    candidate = dict(reference)
    candidate["mask_decoder.head.weight"] = torch.zeros(5, 5)

    result = compare(candidate, reference)
    assert "mask_decoder" not in result["sections"]
    assert result["shared"] == 4


def test_disjoint_checkpoints_report_nothing_shared():
    result = compare({"a.weight": torch.zeros(2)}, {"b.weight": torch.zeros(2)})
    assert result == {"shared": 0, "identical": 0, "sections": {}}


def test_section_summary_is_readable():
    text = summarise_sections({"image_encoder": 0.0, "mask_decoder": 0.125})
    assert "image_encoder" in text and "mask_decoder" in text
    assert "1.250e-01" in text
