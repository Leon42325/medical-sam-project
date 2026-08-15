"""Tests for dataset adapters and the prepare stage, on a synthetic CHAOS tree.

The CHAOS layout is reproduced exactly - CT with binary liver masks and
filenames that carry no slice reference, MR with four organs and annotations
named after their slice - because those two conventions are what the pairing
logic has to get right, and a mistake there is invisible in every later stage.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from samed.cli import prepare as prepare_cli
from samed.data.adapters import available, create
from samed.data.adapters.chaos import CT_TARGETS, MR_TARGETS, pair_slices


def _dicom(path: Path, pixels: np.ndarray, *, instance: int, z: float | None) -> Path:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = -1024
    dataset.InstanceNumber = instance
    if z is not None:
        dataset.ImagePositionPatient = [0.0, 0.0, z]
    dataset.PixelData = pixels.astype(np.int16).tobytes()

    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_as(str(path), enforce_file_format=True)
    return path


def _disc(shape: tuple[int, int], centre: tuple[int, int], radius: int) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.hypot(yy - centre[0], xx - centre[1]) <= radius


@pytest.fixture
def chaos(tmp_path: Path) -> Path:
    """Two CT patients and one MR patient with both sequences."""
    root = tmp_path / "chaos" / "Train_Sets"
    shape = (48, 48)

    for patient, n_slices in (("1", 6), ("10", 5)):
        base = root / "CT" / patient
        for index in range(n_slices):
            # Filenames deliberately do not encode slice order; the header does.
            # z DESCENDS as the filename index ascends, so anatomical order
            # and filename order are opposites. The real CHAOS CT series behave
            # this way, and a fixture where they agree cannot catch a mispairing.
            _dicom(base / "DICOM_anon" / f"i{index:04d},0000b.dcm",
                   np.full(shape, 200 + index * 10, np.int16),
                   instance=index + 1, z=-float(index) * 2.5)
            label = np.zeros(shape, np.uint8)
            label[_disc(shape, (24, 24), 8)] = 255
            cv2.imwrite(str(_mkdir(base / "Ground") / f"liver_GT_{index:03d}.png"), label)

    mr = root / "MR" / "1"
    for sequence in ("T1DUAL", "T2SPIR"):
        dicom_dir = mr / sequence / "DICOM_anon"
        if sequence == "T1DUAL":
            in_phase, out_phase = dicom_dir / "InPhase", dicom_dir / "OutPhase"
        else:
            in_phase, out_phase = dicom_dir, None

        for index in range(4):
            stem = f"IMG-0004-{index:05d}"
            _dicom(in_phase / f"{stem}.dcm", np.full(shape, 100 + index, np.int16),
                   instance=index + 1, z=float(index))
            if out_phase is not None:
                _dicom(out_phase / f"{stem}.dcm", np.full(shape, 300 + index, np.int16),
                       instance=index + 1, z=float(index))

            label = np.zeros(shape, np.uint8)
            label[_disc(shape, (14, 14), 7)] = 63     # liver
            label[_disc(shape, (34, 14), 6)] = 126    # right kidney
            label[_disc(shape, (34, 34), 6)] = 189    # left kidney
            label[_disc(shape, (14, 34), 5)] = 252    # spleen
            cv2.imwrite(str(_mkdir(mr / sequence / "Ground") / f"{stem}.png"), label)

    return tmp_path / "chaos"


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #


def test_chaos_is_registered():
    assert "chaos" in available()


def test_series_cover_every_patient_and_sequence(chaos):
    series = list(create("chaos").series(chaos))
    assert {s.modality for s in series} == {"CT", "T1W-MRI", "T2W-MRI"}
    assert sum(s.modality == "CT" for s in series) == 2
    assert {s.subject for s in series if s.modality != "CT"} == {
        "MR-1-T1DUAL", "MR-1-T2SPIR"
    }


def test_patients_sort_numerically(chaos):
    """Patient 10 must not sort before patient 2."""
    ct = [s.subject for s in create("chaos").series(chaos) if s.modality == "CT"]
    assert ct == ["CT-1", "CT-10"]


def test_ct_and_mr_carry_their_own_label_encodings(chaos):
    series = {s.modality: s for s in create("chaos").series(chaos)}
    assert series["CT"].targets == CT_TARGETS
    assert series["T1W-MRI"].targets == MR_TARGETS
    assert len(series["T1W-MRI"].targets) == 4


def test_outphase_is_dropped(chaos):
    """Both phases share one annotation; keeping both would double-count it."""
    t1 = next(s for s in create("chaos").series(chaos) if s.modality == "T1W-MRI")
    assert len(t1.images) == 4
    assert all("InPhase" in str(p) for p in t1.images)
    assert not any("OutPhase" in str(p) for p in t1.images)


def test_slices_are_ordered_by_filename_index(chaos):
    """CHAOS aligns i0042.dcm with liver_GT_042.png; the numbering is the pairing.

    Ordering by ImagePositionPatient instead looks safer and is wrong: it
    reorders the series and silently pairs every slice with someone else's
    annotation. Measured on the real data, mean liver HU was 144.8 / 143.8 /
    100.2 under filename pairing (portal-venous liver is ~100-140) against
    -87.7 / 59.2 / 11.2 under z-position pairing.
    """
    ct = next(s for s in create("chaos").series(chaos) if s.modality == "CT")
    assert [p.name for p in ct.images] == [f"i{i:04d},0000b.dcm" for i in range(6)]
    assert [p.name for p in ct.labels] == [f"liver_GT_{i:03d}.png" for i in range(6)]

    import pydicom

    positions = [
        float(pydicom.dcmread(str(p), stop_before_pixels=True).ImagePositionPatient[2])
        for p in ct.images
    ]
    assert positions == sorted(positions, reverse=True), (
        "the fixture must have anatomical order opposed to filename order, "
        "otherwise this test cannot distinguish the two"
    )


def test_mr_annotations_are_paired_by_name(chaos):
    t1 = next(s for s in create("chaos").series(chaos) if s.modality == "T1W-MRI")
    assert [i.stem for i in t1.images] == [l.stem for l in t1.labels]


def test_ct_annotations_are_paired_by_position(chaos):
    ct = next(s for s in create("chaos").series(chaos) if s.modality == "CT")
    assert [l.name for l in ct.labels] == [f"liver_GT_{i:03d}.png" for i in range(6)]


def test_pairing_refuses_a_length_mismatch(tmp_path):
    images = [tmp_path / f"i{i}.dcm" for i in range(3)]
    labels = [tmp_path / "liver_GT_000.png"]
    with pytest.raises(ValueError, match="cannot pair"):
        pair_slices(images, labels)


def test_series_rejects_unequal_images_and_labels(tmp_path):
    from samed.data.adapters import Series

    with pytest.raises(ValueError, match="not one-to-one"):
        Series("d", "CT", "p1", [tmp_path / "a.dcm"], [])


def test_missing_download_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="CHAOS download"):
        list(create("chaos").series(tmp_path))


# --------------------------------------------------------------------------- #
# Prepare
# --------------------------------------------------------------------------- #


def _run_prepare(chaos: Path, out: Path, **overrides) -> list[dict]:
    argv = ["--dataset", "chaos", "--root", str(chaos), "--out", str(out)]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    assert prepare_cli.main(argv) == 0
    with (out / "manifest-chaos.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_prepare_emits_one_row_per_object_instance(chaos, tmp_path):
    rows = _run_prepare(chaos, tmp_path / "prepared")

    # CT: 11 slices x 1 target. MR: 4 slices x 4 targets x 2 sequences.
    assert sum(r["modality"] == "CT" for r in rows) == 11
    assert sum(r["modality"] == "T1W-MRI" for r in rows) == 16
    assert {r["target"] for r in rows if r["modality"] == "T1W-MRI"} == set(MR_TARGETS)


def test_prepare_writes_normalised_uint8_slices(chaos, tmp_path):
    out = tmp_path / "prepared"
    _run_prepare(chaos, out)
    for path in (out / "images").rglob("*.png"):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image.dtype == np.uint8 and image.ndim == 2


def test_prepare_writes_one_image_per_slice_not_per_target(chaos, tmp_path):
    """Four MR targets share one slice; the pixels must be written once."""
    out = tmp_path / "prepared"
    rows = _run_prepare(chaos, out)
    mr_rows = [r for r in rows if r["modality"] == "T1W-MRI"]
    assert len({r["image_id"] for r in mr_rows}) == 4
    assert len(mr_rows) == 16


def test_prepare_keeps_label_values_intact(chaos, tmp_path):
    out = tmp_path / "prepared"
    rows = _run_prepare(chaos, out)
    row = next(r for r in rows if r["target"] == "spleen")
    label = cv2.imread(str(out / "labels" / row["label_path"]), cv2.IMREAD_UNCHANGED)
    assert int(row["label_value"]) == 252
    assert (label == 252).sum() > 0


def test_prepare_drops_targets_below_the_area_threshold(chaos, tmp_path):
    """Fixture areas: liver 197 px, kidneys 113 px, spleen 81 px.

    A threshold of 150 must keep the liver and drop the rest - this is the rule
    that removes the partial-volume slices at an organ's extremities.
    """
    rows = _run_prepare(chaos, tmp_path / "big-threshold", min_label_area=150)
    assert {r["target"] for r in rows} == {"liver"}


def test_prepare_honours_the_sampling_budget(chaos, tmp_path):
    rows = _run_prepare(chaos, tmp_path / "capped", max_per_target=2)
    for modality in {r["modality"] for r in rows}:
        for target in {r["target"] for r in rows if r["modality"] == modality}:
            count = sum(
                1 for r in rows if r["modality"] == modality and r["target"] == target
            )
            assert count <= 2


def test_prepare_writes_overlays_for_visual_qc(chaos, tmp_path):
    out = tmp_path / "prepared"
    _run_prepare(chaos, out, save_overlays=2)
    overlays = sorted((out / "overlays").glob("*.png"))
    assert overlays
    sample = cv2.imread(str(overlays[0]), cv2.IMREAD_UNCHANGED)
    assert sample.ndim == 3, "an overlay is colour, so the annotation is visible"


def test_prepare_rejects_a_size_mismatch_between_slice_and_annotation(chaos, tmp_path):
    ground = next((chaos / "Train_Sets" / "CT" / "1" / "Ground").glob("*.png"))
    cv2.imwrite(str(ground), np.zeros((16, 16), np.uint8))
    with pytest.raises(ValueError, match="but its annotation"):
        _run_prepare(chaos, tmp_path / "bad")


def test_overlay_gives_every_label_value_a_visible_colour():
    """A label of 255 used to map to pure black, hiding the annotation entirely.

    That is what made a CT mispairing hard to see in the overlays - the check
    that exists precisely to reveal it.
    """
    from samed.cli.prepare import _overlay

    image = np.full((16, 16), 120, np.uint8)
    for value in (1, 2, 3, 63, 126, 189, 252, 255):
        label = np.zeros((16, 16), np.uint8)
        label[4:12, 4:12] = value
        tinted = _overlay(image, label)[8, 8]
        assert tinted.max() > 60, f"label {value} is invisible: BGR {tinted}"
        assert not np.array_equal(tinted, image[8, 8]), f"label {value} left no tint"


def test_overlays_are_spread_over_the_organ_not_taken_from_its_tip(chaos, tmp_path):
    """Overlays exist to reveal a slice-to-annotation mismatch.

    The first slices to clear the area threshold are the slivers at the very
    edge of an organ, where a mask is small and ambiguous and a misalignment is
    almost impossible to see. Sampling across the kept range is what puts the
    large mid-organ sections in front of a human.
    """
    out = tmp_path / "prepared"
    label = np.zeros((48, 48), np.uint8)
    label[_disc((48, 48), (24, 24), 8)] = 255

    # Give one CT patient a long run of annotated slices, so "first three" and
    # "spread over the range" are visibly different choices.
    base = chaos / "Train_Sets" / "CT" / "1"
    for index in range(6, 20):
        _dicom(base / "DICOM_anon" / f"i{index:04d},0000b.dcm",
               np.full((48, 48), 200 + index, np.int16), instance=index + 1,
               z=-float(index) * 2.5)
        cv2.imwrite(str(base / "Ground" / f"liver_GT_{index:03d}.png"), label)

    _run_prepare(chaos, out, save_overlays=3)
    indices = sorted(
        int(p.stem.rsplit("_", 1)[1])
        for p in (out / "overlays").glob("chaos_CT_CT-1_*.png")
    )
    assert len(indices) == 3
    assert max(indices) - min(indices) >= 8, (
        f"overlays {indices} are clustered; they must span the annotated range"
    )
