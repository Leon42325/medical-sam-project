"""Integrity checks a downloaded dataset must pass before it is used.

Rule 3 of the provenance policy in ``configs/datasets.yaml`` says a third-party
copy may be used only after verification. This module is that verification.

The check that matters most is resolution. Re-uploads are routinely resized, and
a resized copy silently corrupts this study specifically: object size is measured
in pixels and boundary complexity is an elliptic Fourier order fitted to a
rasterised contour, so a 565x584 fundus image downsampled to 256x256 does not
have the same attributes as the original - and attributes are the independent
variables of H1 and H2. A resized mirror would not crash anything; it would just
quietly answer a different question.

Label values matter for the same reason at a coarser level: a mirror that has
binarised a multi-class label map has thrown away every target but one.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "Check",
    "VerificationReport",
    "IMAGE_SUFFIXES",
    "sha256_of",
    "read_sizes",
    "inspect_images",
    "verify_dataset",
]

#: Raster formats readable with OpenCV. These are what the paper's preprocessing
#: (Sec. 2.2) converts everything *to*; label maps usually arrive as these too.
RASTER_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})

#: Formats the raw datasets actually ship in. CHAOS distributes DICOM series,
#: MSD distributes NIfTI volumes; neither is readable with OpenCV, and treating
#: them as "not an image" is how a correctly downloaded dataset gets reported as
#: empty.
DICOM_SUFFIXES = frozenset({".dcm", ".ima"})
VOLUME_SUFFIXES = frozenset({".nii", ".gz", ".mha", ".mhd", ".nrrd"})

IMAGE_SUFFIXES = RASTER_SUFFIXES | DICOM_SUFFIXES | VOLUME_SUFFIXES

#: Sizes that essentially never occur natively in medical imaging but are the
#: standard outputs of a resizing pipeline. Their presence is not proof, but it
#: warrants a look before the data is trusted.
_SUSPICIOUS_SQUARE_SIDES = frozenset({128, 224, 256, 384, 512})


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    expected: str
    actual: str
    fatal: bool = True

    def render(self) -> str:
        mark = "ok  " if self.passed else ("FAIL" if self.fatal else "warn")
        return f"  [{mark}] {self.name}: expected {self.expected}, got {self.actual}"


@dataclass
class VerificationReport:
    dataset: str
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no fatal check failed; warnings do not block use."""
        return all(check.passed or not check.fatal for check in self.checks)

    def add(self, name: str, passed: bool, expected, actual, *, fatal: bool = True) -> None:
        self.checks.append(Check(name, passed, str(expected), str(actual), fatal))

    def render(self) -> str:
        head = f"{self.dataset}: {'PASS' if self.ok else 'REJECTED'}"
        return "\n".join([head, *(c.render() for c in self.checks)])


def sha256_of(path: str | Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_sizes(path: Path) -> list[tuple[int, int]]:
    """In-plane ``(width, height)`` of every slice in a file.

    A raster or DICOM file yields one entry; a NIfTI volume yields one per
    slice, so that a 3D dataset is measured in the same units as a 2D one.
    Headers are read without decoding pixel data where the format allows it.
    """
    suffix = "".join(path.suffixes[-2:]).lower() if path.name.endswith(".gz") else path.suffix.lower()

    if suffix in DICOM_SUFFIXES:
        try:
            import pydicom
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"{path.name} is DICOM; install pydicom to verify this dataset"
            ) from error
        header = pydicom.dcmread(str(path), stop_before_pixels=True)
        return [(int(header.Columns), int(header.Rows))]

    if suffix in {".nii", ".nii.gz"} or path.suffix.lower() in VOLUME_SUFFIXES:
        try:
            import nibabel
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"{path.name} is a volume; install nibabel to verify this dataset"
            ) from error
        shape = nibabel.load(str(path)).header.get_data_shape()
        if len(shape) < 3:
            return [(int(shape[0]), int(shape[1]))]
        return [(int(shape[0]), int(shape[1]))] * int(shape[2])

    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return []
    return [(image.shape[1], image.shape[0])]


def inspect_images(paths: Iterable[str | Path]) -> tuple[Counter, Counter]:
    """Return counters of ``(width, height)`` sizes and of pixel values seen.

    Pixel values are collected only from single-channel raster files, which is
    what label maps are in every dataset here; volumes and colour images
    contribute sizes only. Reading label values is the point of the second
    counter, and it needs actual pixels, so it stays deliberately narrow.
    """
    import cv2
    import numpy as np

    sizes: Counter = Counter()
    values: Counter = Counter()
    for raw in paths:
        path = Path(raw)
        sizes.update(read_sizes(path))
        if path.suffix.lower() in RASTER_SUFFIXES:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is not None and image.ndim == 2:
                values.update(np.unique(image).tolist())
    return sizes, values


def _check_label_group(
    report: VerificationReport,
    label: str,
    paths: Sequence[str | Path],
    expected: Sequence[int] | None,
) -> None:
    if not paths:
        report.add(f"labels present ({label})", False, "> 0 files", 0)
        return

    _, values = inspect_images(paths)
    found = sorted(int(v) for v in values if v != 0)
    if expected is None:
        report.add(
            f"labels are not binarised ({label})",
            len(found) > 0,
            "at least one non-zero category",
            found,
            fatal=False,
        )
    else:
        report.add(f"label values ({label})", found == sorted(int(v) for v in expected),
                   sorted(int(v) for v in expected), found)


def verify_dataset(
    name: str,
    *,
    image_paths: Sequence[str | Path],
    label_paths: Sequence[str | Path] | None = None,
    expect_count: int | None = None,
    expect_resolutions: Sequence[Sequence[int]] | None = None,
    expect_label_values: Sequence[int] | dict[str, Sequence[int]] | None = None,
    root: str | Path | None = None,
) -> VerificationReport:
    """Check a downloaded dataset against what its source publication claims.

    ``expect_resolutions`` is a list of allowed ``(width, height)`` pairs. Leave
    it out when the native resolution varies per image (ISIC, Kvasir-SEG); the
    resizing heuristic still runs and will flag a mirror where every image
    happens to share one square size.

    ``expect_label_values`` is a flat list when one encoding covers the dataset,
    or a mapping of glob pattern to expected values when it does not - which is
    not an edge case. CHAOS annotates the liver alone in CT, as a binary mask,
    but four organs in MR at 63/126/189/252; a single flat expectation is
    guaranteed to be wrong for one of the two. ``root`` is needed to resolve
    those patterns and is only required in that case.
    """
    report = VerificationReport(dataset=name)

    report.add("files present", len(image_paths) > 0, "> 0", len(image_paths))
    if not image_paths:
        return report

    if expect_count is not None:
        report.add("image count", len(image_paths) == expect_count, expect_count, len(image_paths))

    sizes, _ = inspect_images(image_paths)
    observed = sorted(sizes)

    if expect_resolutions is not None:
        allowed = {tuple(r) for r in expect_resolutions}
        unexpected = [s for s in observed if s not in allowed]
        report.add(
            "native resolution",
            not unexpected,
            sorted(allowed),
            f"{observed[:3]}{'...' if len(observed) > 3 else ''}",
        )
    elif len(observed) == 1:
        width, height = observed[0]
        looks_resized = width == height and width in _SUSPICIOUS_SQUARE_SIDES
        report.add(
            "not uniformly resized",
            not looks_resized,
            "varying sizes, or a non-standard uniform size",
            f"every image is {width}x{height}",
            fatal=False,
        )

    if label_paths:
        if isinstance(expect_label_values, dict):
            if root is None:
                raise ValueError("per-pattern label expectations need `root` to resolve them")
            remaining = set(Path(p) for p in label_paths)
            for pattern, expected in expect_label_values.items():
                group = sorted(remaining & set(Path(root).glob(pattern)))
                remaining -= set(group)
                _check_label_group(report, pattern, group, expected)
            if remaining:
                # Files no pattern claimed are a gap in the expectation, not a
                # silent pass: they would be evaluated with an unknown encoding.
                report.add("every label file is accounted for", False,
                           "0 unmatched", f"{len(remaining)} unmatched")
        else:
            _check_label_group(report, "all", label_paths, expect_label_values)

    return report
