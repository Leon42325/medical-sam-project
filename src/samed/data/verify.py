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

__all__ = ["Check", "VerificationReport", "sha256_of", "inspect_images", "verify_dataset"]

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


def inspect_images(paths: Iterable[str | Path]) -> tuple[Counter, Counter]:
    """Return counters of ``(width, height)`` sizes and of pixel values seen.

    Pixel values are collected only for single-channel images, which is what
    label maps are; colour images contribute sizes only.
    """
    import cv2
    import numpy as np

    sizes: Counter = Counter()
    values: Counter = Counter()
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        height, width = image.shape[:2]
        sizes[(width, height)] += 1
        if image.ndim == 2:
            values.update(np.unique(image).tolist())
    return sizes, values


def verify_dataset(
    name: str,
    *,
    image_paths: Sequence[str | Path],
    label_paths: Sequence[str | Path] | None = None,
    expect_count: int | None = None,
    expect_resolutions: Sequence[Sequence[int]] | None = None,
    expect_label_values: Sequence[int] | None = None,
) -> VerificationReport:
    """Check a downloaded dataset against what its source publication claims.

    ``expect_resolutions`` is a list of allowed ``(width, height)`` pairs. Leave
    it out when the native resolution varies per image (ISIC, Kvasir-SEG); the
    resizing heuristic still runs and will flag a mirror where every image
    happens to share one square size.
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
        _, values = inspect_images(label_paths)
        found = sorted(int(v) for v in values if v != 0)
        if expect_label_values is not None:
            expected = sorted(int(v) for v in expect_label_values)
            report.add("label values", found == expected, expected, found)
        else:
            report.add(
                "labels are not binarised",
                len(found) > 0,
                "at least one non-zero category",
                found,
                fatal=False,
            )

    return report
