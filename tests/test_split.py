"""Tests for the patient-level split that the fine-tuning experiment rests on."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from samed.cli import split as split_cli
from samed.data.manifest import ManifestRow, write_manifest
from samed.data.split import coverage_gaps, describe_split, split_by_patient


def _rows(n_ct: int = 20, n_mr: int = 20, slices: int = 30) -> list[ManifestRow]:
    """A CHAOS-shaped manifest: CT patients with a liver, MR patients with four
    organs across two sequences."""
    rows = []
    for patient in range(n_ct):
        for index in range(slices):
            rows.append(ManifestRow(
                dataset="chaos", modality="CT", target="liver",
                subject=f"CT-{patient}", patient=f"CT-{patient}",
                image_id=f"ct{patient}_{index}", image_path="i.png",
                label_path="l.png", label_value=255, slice_index=index,
            ))
    for patient in range(n_mr):
        for sequence, modality in (("T1DUAL", "T1W-MRI"), ("T2SPIR", "T2W-MRI")):
            for target, value in (("liver", 63), ("right kidney", 126),
                                  ("left kidney", 189), ("spleen", 252)):
                for index in range(slices):
                    rows.append(ManifestRow(
                        dataset="chaos", modality=modality, target=target,
                        subject=f"MR-{patient}-{sequence}", patient=f"MR-{patient}",
                        image_id=f"mr{patient}_{sequence}_{index}", image_path="i.png",
                        label_path="l.png", label_value=value, slice_index=index,
                    ))
    return rows


def test_no_patient_appears_in_two_splits():
    """The failure this module exists to prevent: slices of one scan on both
    sides measure memorisation and report it as generalisation."""
    splits = split_by_patient(_rows())
    owners = {}
    for name, rows in splits.items():
        for row in rows:
            assert owners.setdefault(row.patient, name) == name


def test_every_target_survives_in_every_split():
    assert coverage_gaps(split_by_patient(_rows())) == {}


def test_a_patients_slices_stay_together():
    splits = split_by_patient(_rows())
    for name, rows in splits.items():
        for patient in {r.patient for r in rows}:
            mine = sum(1 for r in rows if r.patient == patient)
            elsewhere = sum(
                1 for other, rest in splits.items() if other != name
                for r in rest if r.patient == patient
            )
            assert elsewhere == 0 and mine > 0


def test_both_mr_sequences_of_one_person_stay_together():
    """T1DUAL and T2SPIR of MR patient 3 are the same body."""
    splits = split_by_patient(_rows())
    for name, rows in splits.items():
        subjects = {r.subject for r in rows if r.patient == "MR-3"}
        assert subjects in ({"MR-3-T1DUAL", "MR-3-T2SPIR"}, set())


def test_the_split_is_deterministic():
    first = split_by_patient(_rows(), seed=5)
    second = split_by_patient(_rows(), seed=5)
    for name in first:
        assert [r.image_id for r in first[name]] == [r.image_id for r in second[name]]


def test_a_different_seed_moves_patients():
    a = {r.patient for r in split_by_patient(_rows(), seed=1)["test"]}
    b = {r.patient for r in split_by_patient(_rows(), seed=2)["test"]}
    assert a != b


def test_fractions_are_respected_within_each_modality():
    splits = split_by_patient(_rows(n_ct=20, n_mr=20))
    for modality in ("CT", "T1W-MRI"):
        patients = {
            name: len({r.patient for r in rows if r.modality == modality})
            for name, rows in splits.items()
        }
        assert patients["train"] == 12 and patients["val"] == 3 and patients["test"] == 5


def test_no_split_is_left_empty_in_a_small_stratum():
    """Largest-remainder rounding would give val zero patients out of five."""
    splits = split_by_patient(_rows(n_ct=5, n_mr=5))
    for name, rows in splits.items():
        assert rows, f"{name} is empty"
    assert coverage_gaps(splits) == {}


def test_too_few_patients_fails_loudly():
    with pytest.raises(ValueError, match="cannot fill 3 splits from 2"):
        split_by_patient(_rows(n_ct=2, n_mr=2))


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1"):
        split_by_patient(_rows(), fractions={"train": 0.5, "test": 0.2})


def test_description_reports_patients_not_only_rows():
    text = describe_split(split_by_patient(_rows()))
    assert "patients" in text and "train" in text


def test_cli_writes_one_manifest_per_split(tmp_path: Path):
    manifest = tmp_path / "manifest-chaos.csv"
    write_manifest(manifest, _rows())

    assert split_cli.main(["--manifest", str(manifest)]) == 0

    for name in ("train", "val", "test"):
        path = tmp_path / f"manifest-chaos-{name}.csv"
        assert path.exists()
        with path.open(newline="") as handle:
            assert list(csv.DictReader(handle))


def test_cli_refuses_a_split_that_drops_a_target(tmp_path: Path, capsys):
    """One CT patient cannot supply train, val and test."""
    manifest = tmp_path / "manifest-tiny.csv"
    write_manifest(manifest, _rows(n_ct=1, n_mr=20))

    with pytest.raises(ValueError, match="cannot fill"):
        split_cli.main(["--manifest", str(manifest)])
