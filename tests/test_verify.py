"""Tests for dataset verification - the enforcement of provenance policy rule 3."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from samed.cli import fetch as fetch_cli
from samed.data.verify import inspect_images, sha256_of, verify_dataset


def _write(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), array)
    return path


def _native_images(root: Path, count: int = 4) -> list[Path]:
    """Images at DRIVE's native 565x584, as an untouched copy would be."""
    return [
        _write(root / f"img{i}.png", np.random.default_rng(i).integers(0, 255, (584, 565), np.uint8))
        for i in range(count)
    ]


def _resized_images(root: Path, count: int = 4, side: int = 256) -> list[Path]:
    return [
        _write(root / f"img{i}.png",
               np.random.default_rng(i).integers(0, 255, (side, side), np.uint8))
        for i in range(count)
    ]


# --------------------------------------------------------------------------- #


def test_sha256_is_stable_and_content_dependent(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"medical images")
    b.write_bytes(b"medical images")
    assert sha256_of(a) == sha256_of(b)
    b.write_bytes(b"different")
    assert sha256_of(a) != sha256_of(b)


def test_inspect_images_reports_sizes_and_label_values(tmp_path):
    _write(tmp_path / "a.png", np.zeros((10, 20), np.uint8))
    label = np.zeros((10, 20), np.uint8)
    label[2:5, 2:5] = 63
    label[6:8, 6:8] = 126
    _write(tmp_path / "b.png", label)

    sizes, values = inspect_images(sorted(tmp_path.glob("*.png")))
    assert sizes == {(20, 10): 2}
    assert set(values) == {0, 63, 126}


def test_a_native_copy_passes(tmp_path):
    report = verify_dataset(
        "drive", image_paths=_native_images(tmp_path),
        expect_count=4, expect_resolutions=[[565, 584]],
    )
    assert report.ok, report.render()


def test_a_resized_mirror_is_rejected(tmp_path):
    """The failure this module exists to catch."""
    report = verify_dataset(
        "drive", image_paths=_resized_images(tmp_path),
        expect_count=4, expect_resolutions=[[565, 584]],
    )
    assert not report.ok
    failures = [c.name for c in report.checks if not c.passed]
    assert "native resolution" in failures


def test_a_wrong_image_count_is_rejected(tmp_path):
    report = verify_dataset("isic2018", image_paths=_native_images(tmp_path, 3), expect_count=2594)
    assert not report.ok
    assert any(c.name == "image count" and not c.passed for c in report.checks)


def test_uniform_square_sizes_are_flagged_when_no_resolution_is_declared(tmp_path):
    """For datasets with naturally varying sizes, uniformity is the tell."""
    report = verify_dataset("mirror", image_paths=_resized_images(tmp_path))
    warning = next(c for c in report.checks if c.name == "not uniformly resized")
    assert not warning.passed
    assert not warning.fatal, "a heuristic warns, it does not reject"
    assert report.ok


def test_varying_sizes_do_not_trigger_the_heuristic(tmp_path):
    paths = [
        _write(tmp_path / "a.png", np.zeros((600, 800), np.uint8)),
        _write(tmp_path / "b.png", np.zeros((480, 640), np.uint8)),
    ]
    report = verify_dataset("kvasir", image_paths=paths)
    assert report.ok
    assert not any(c.name == "not uniformly resized" for c in report.checks)


def test_re_encoded_label_values_are_rejected(tmp_path):
    """A mirror that binarised a multi-class label map has lost every target but one."""
    images = _native_images(tmp_path / "images", 2)
    binarised = np.zeros((584, 565), np.uint8)
    binarised[10:20, 10:20] = 255
    labels = [_write(tmp_path / "masks" / "a.png", binarised)]

    report = verify_dataset(
        "chaos", image_paths=images, label_paths=labels,
        expect_label_values=[63, 126, 189, 252],
    )
    assert not report.ok
    assert any(c.name.startswith("label values") and not c.passed for c in report.checks)


def test_correct_label_values_pass(tmp_path):
    images = _native_images(tmp_path / "images", 2)
    label = np.zeros((584, 565), np.uint8)
    for offset, value in enumerate([63, 126, 189, 252]):
        label[10 + offset * 12 : 20 + offset * 12, 10:20] = value
    labels = [_write(tmp_path / "masks" / "a.png", label)]

    report = verify_dataset(
        "chaos", image_paths=images, label_paths=labels,
        expect_label_values=[63, 126, 189, 252],
    )
    assert report.ok, report.render()


def test_an_empty_directory_is_rejected():
    report = verify_dataset("nothing", image_paths=[])
    assert not report.ok
    assert report.checks[0].name == "files present"


def test_report_renders_the_verdict(tmp_path):
    text = verify_dataset("drive", image_paths=_resized_images(tmp_path),
                          expect_resolutions=[[565, 584]]).render()
    assert "REJECTED" in text and "native resolution" in text


# --------------------------------------------------------------------------- #
# The fetch CLI's local behaviour (no network)
# --------------------------------------------------------------------------- #


def test_find_images_separates_labels_from_images(tmp_path):
    _write(tmp_path / "images" / "a.png", np.zeros((8, 8), np.uint8))
    _write(tmp_path / "masks" / "a.png", np.zeros((8, 8), np.uint8))
    _write(tmp_path / "ground_truth" / "b.png", np.zeros((8, 8), np.uint8))

    images, labels = fetch_cli.find_images(tmp_path)
    assert [p.name for p in images] == ["a.png"]
    assert sorted(p.parent.name for p in labels) == ["ground_truth", "masks"]


def test_every_configured_source_is_well_formed():
    """The config drives the fetcher, so a typo there is a silent wrong download."""
    import yaml

    sources = yaml.safe_load(fetch_cli.CONFIG.read_text())["sources"]
    assert sources, "configs/sources.yaml must not be empty"

    for name, spec in sources.items():
        kind = spec.get("type")
        assert kind in {"zenodo", "kaggle", "manual"}, f"{name}: bad type {kind!r}"
        if kind == "zenodo":
            assert isinstance(spec["record"], int), f"{name}: record must be a Zenodo id"
        elif kind == "kaggle":
            assert "/" in spec["slug"], f"{name}: slug must be owner/dataset"
        else:
            assert spec["url"].startswith("http"), f"{name}: manual source needs a URL"


def test_unknown_dataset_is_reported(capsys):
    assert fetch_cli.main(["--dataset", "not_a_dataset"]) == 2
    assert "unknown dataset" in capsys.readouterr().out


def test_manual_sources_print_instructions_instead_of_guessing(tmp_path, capsys):
    assert fetch_cli.main(["--dataset", "camus", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "manual download required" in out
    assert "humanheart-project" in out


def test_dry_run_downloads_nothing(tmp_path, capsys):
    assert fetch_cli.main(["--dataset", "isic2018", "--root", str(tmp_path), "--dry-run"]) == 0
    assert "would run: kaggle datasets download" in capsys.readouterr().out
    assert not any(tmp_path.rglob("*.zip"))


def test_a_failing_source_does_not_abandon_the_others(tmp_path, capsys, monkeypatch):
    """One unreachable host must cost one dataset, not the whole run."""
    def explode(spec, target, *, dry_run):
        raise RuntimeError("could not reach zenodo.org: nope")

    monkeypatch.setattr(fetch_cli, "fetch_zenodo", explode)
    assert fetch_cli.main(["--dataset", "chaos", "--root", str(tmp_path)]) == 1

    out = capsys.readouterr().out
    assert "ERROR could not reach" in out
    assert "rejected: chaos" in out


# --------------------------------------------------------------------------- #
# The formats the raw datasets actually ship in
# --------------------------------------------------------------------------- #


def test_read_sizes_handles_nifti_volumes(tmp_path):
    """MSD ships NIfTI; a volume must be measured per slice, like 2D data is."""
    import nibabel as nib

    volume = np.zeros((320, 260, 7), dtype=np.int16)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), str(tmp_path / "liver.nii.gz"))

    from samed.data.verify import read_sizes

    sizes = read_sizes(tmp_path / "liver.nii.gz")
    assert sizes == [(320, 260)] * 7


def test_read_sizes_handles_dicom(tmp_path):
    """CHAOS ships DICOM series; sizes come from the header, not the pixel data."""
    import pydicom
    from pydicom.dataset import FileMetaDataset

    meta = FileMetaDataset()
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()

    dataset = pydicom.dataset.FileDataset(
        "slice.dcm", {}, file_meta=meta, preamble=b"\0" * 128
    )
    dataset.Rows, dataset.Columns = 512, 400
    path = tmp_path / "slice.dcm"
    dataset.save_as(str(path), enforce_file_format=True)

    from samed.data.verify import read_sizes

    assert read_sizes(path) == [(400, 512)]


def test_dicom_and_nifti_count_as_images():
    from samed.data.verify import IMAGE_SUFFIXES

    assert {".dcm", ".nii"} <= IMAGE_SUFFIXES


def test_explicit_layout_beats_the_keyword_heuristic(tmp_path):
    """CHAOS is the case that broke the heuristic: DICOM images, PNG labels
    living under a directory whose name contains 'ground'."""
    (tmp_path / "Train_Sets" / "CT" / "1" / "DICOM_anon").mkdir(parents=True)
    (tmp_path / "Train_Sets" / "CT" / "1" / "Ground").mkdir(parents=True)
    (tmp_path / "Train_Sets" / "CT" / "1" / "DICOM_anon" / "i0001.dcm").write_bytes(b"x")
    _write(tmp_path / "Train_Sets" / "CT" / "1" / "Ground" / "l0001.png",
           np.zeros((8, 8), np.uint8))

    heuristic_images, _ = fetch_cli.find_images(tmp_path)
    assert [p.name for p in heuristic_images] == ["i0001.dcm"]

    images, labels = fetch_cli.find_images(tmp_path, {
        "images": "Train_Sets/*/*/DICOM_anon/**/*.dcm",
        "labels": "Train_Sets/*/*/Ground/*.png",
    })
    assert [p.name for p in images] == ["i0001.dcm"]
    assert [p.name for p in labels] == ["l0001.png"]


def test_one_dataset_can_carry_two_label_encodings(tmp_path):
    """CHAOS: liver-only binary masks in CT, four organs at 63/126/189/252 in MR.

    A single flat expectation would fail one arm no matter which one it stated.
    """
    ct = tmp_path / "Train_Sets" / "CT" / "1" / "Ground"
    mr = tmp_path / "Train_Sets" / "MR" / "1" / "T1DUAL" / "Ground"

    binary = np.zeros((64, 64), np.uint8)
    binary[10:20, 10:20] = 255
    _write(ct / "liver_GT_000.png", binary)

    multi = np.zeros((64, 64), np.uint8)
    for offset, value in enumerate([63, 126, 189, 252]):
        multi[10 + offset * 12 : 20 + offset * 12, 10:20] = value
    _write(mr / "IMG-0000.png", multi)

    labels = sorted(tmp_path.rglob("Ground/*.png"))
    report = verify_dataset(
        "chaos", image_paths=_native_images(tmp_path / "img", 2), label_paths=labels,
        expect_label_values={
            "Train_Sets/CT/**/Ground/*.png": [255],
            "Train_Sets/MR/**/Ground/*.png": [63, 126, 189, 252],
        },
        root=tmp_path,
    )
    assert report.ok, report.render()


def test_label_files_no_pattern_claims_are_reported(tmp_path):
    """An unmatched file would otherwise be evaluated with an unknown encoding."""
    _write(tmp_path / "Train_Sets" / "CT" / "1" / "Ground" / "a.png", np.zeros((8, 8), np.uint8))
    _write(tmp_path / "Elsewhere" / "stray.png", np.zeros((8, 8), np.uint8))

    report = verify_dataset(
        "chaos", image_paths=_native_images(tmp_path / "img", 1),
        label_paths=sorted(tmp_path.rglob("*.png")),
        expect_label_values={"Train_Sets/CT/**/Ground/*.png": None},
        root=tmp_path,
    )
    assert not report.ok
    assert any(c.name == "every label file is accounted for" for c in report.checks)


def test_per_pattern_expectations_need_a_root(tmp_path):
    with pytest.raises(ValueError, match="need `root`"):
        verify_dataset("x", image_paths=_native_images(tmp_path, 1),
                       label_paths=[tmp_path / "img0.png"],
                       expect_label_values={"**/*.png": [1]})
