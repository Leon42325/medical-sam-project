"""CHAOS: Combined (CT-MR) Healthy Abdominal Organ Segmentation.

Layout, as confirmed against the Zenodo release (record 3431873):

    Train_Sets/CT/<patient>/DICOM_anon/*.dcm
    Train_Sets/CT/<patient>/Ground/liver_GT_*.png
    Train_Sets/MR/<patient>/T1DUAL/DICOM_anon/InPhase/*.dcm
    Train_Sets/MR/<patient>/T1DUAL/DICOM_anon/OutPhase/*.dcm
    Train_Sets/MR/<patient>/T1DUAL/Ground/*.png
    Train_Sets/MR/<patient>/T2SPIR/DICOM_anon/*.dcm
    Train_Sets/MR/<patient>/T2SPIR/Ground/*.png

20 CT patients (2874 slices) and 20 MR patients with two sequences each
(1917 slices).

Three decisions worth stating, because each changes what gets measured:

**CT and MR use different label encodings.** CT annotates the liver alone as a
binary mask; MR annotates four organs at 63, 126, 189 and 252. So CHAOS
contributes 1 + 4 + 4 = 9 object-modality targets, with the liver appearing in
all three modalities - which is precisely the comparison that isolates the
modality factor the paper reports as uncorrelated with DICE (Table 6).

**T1DUAL's OutPhase is dropped.** The in-phase and out-of-phase images are
co-registered and share a single Ground directory. Keeping both would count the
same annotation twice, inflating the effective sample size and breaking the
independence the sampling protocol exists to protect. InPhase is kept as the
paper's "T1W MRI".

**Slice order comes from the filename index, not the DICOM header.** CT slices
are ``i0000,0000b.dcm`` … ``i0095,0000b.dcm`` and annotations are
``liver_GT_000.png`` … ``liver_GT_095.png``: the numbering *is* the
correspondence, and CHAOS intends it to be used.

Sorting by ``ImagePositionPatient`` instead - anatomical order, which sounds
safer - reorders the series and destroys that correspondence. It was tried
first, and it was wrong. The evidence, mean CT number inside the liver mask
under each pairing:

    patient   by filename   by z-position
    CT/1          144.8          -87.7
    CT/19         143.8           59.2
    CT/21         100.2           11.2

CHAOS CT is portal-venous contrast-enhanced, where liver reads ~100-140 HU. The
filename pairing is consistent across patients and lands in that range; the
z-position pairing scatters, including a value in the fat/air range that no
liver voxel can produce. Consistency across patients is the discriminating
signal here, not the absolute number.

The failure mode is worth naming, because it is the one this whole pipeline is
built to avoid: a mispairing raises no error, produces plausible-looking output,
and turns every DICE score downstream into noise. It was caught by looking at
image/label overlays, which is why ``prepare`` writes them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from samed.data.adapters import Adapter, Series, register

__all__ = ["ChaosAdapter"]

#: Label pixel values in the MR ground truth, from the challenge documentation
#: and confirmed against the download.
MR_TARGETS = {
    "liver": 63,
    "right kidney": 126,
    "left kidney": 189,
    "spleen": 252,
}

#: CT ground truth is a binary liver mask.
CT_TARGETS = {"liver": 255}

#: Modality names follow the paper's own vocabulary (Table 1) so that results
#: line up with its per-modality reporting.
MODALITY = {"CT": "CT", "T1DUAL": "T1W-MRI", "T2SPIR": "T2W-MRI"}


def _slice_index(path: Path) -> tuple:
    """Sort key following the numbering CHAOS uses to align slices and annotations.

    The leading integer of the filename (``i0042,0000b.dcm`` -> 42,
    ``liver_GT_042.png`` -> 42) is the index both sides share. Numeric rather
    than lexical so that a series without zero padding still orders correctly;
    the name is kept as a tie-break for determinism.
    """
    digits = re.search(r"\d+", path.stem)
    return (0, int(digits.group()), path.name) if digits else (1, 0, path.name)


def pair_slices(images: list[Path], labels: list[Path]) -> tuple[list[Path], list[Path]]:
    """Align an ordered DICOM series with its annotations.

    Two conventions occur in CHAOS. MR annotations carry the same stem as their
    DICOM slice, so they can be matched by name - which is exact, and immune to
    any ordering mistake. CT annotations are named ``liver_GT_000.png`` and
    carry no reference to the slice, leaving positional pairing as the only
    option: the *i*-th annotation belongs to the *i*-th slice in anatomical
    order.

    Name matching is attempted first for exactly that reason, and positional
    pairing is used only when it cannot apply. Both sequences must already be
    ordered by :func:`_slice_index`; this function does not reorder them,
    because the ordering *is* the pairing.
    """
    by_stem = {path.stem: path for path in labels}
    if all(image.stem in by_stem for image in images) and len(by_stem) == len(labels):
        return images, [by_stem[image.stem] for image in images]

    ordered_labels = list(labels)
    if len(images) != len(ordered_labels):
        raise ValueError(
            f"cannot pair {len(images)} slices with {len(ordered_labels)} annotations "
            "by position, and their names do not correspond"
        )
    return images, ordered_labels


class ChaosAdapter(Adapter):
    name = "chaos"

    def series(self, root: Path) -> Iterator[Series]:
        train = root / "Train_Sets"
        if not train.is_dir():
            raise FileNotFoundError(f"expected {train} - is this a CHAOS download?")

        yield from self._ct_series(train / "CT")
        yield from self._mr_series(train / "MR")

    def _ct_series(self, ct_root: Path) -> Iterator[Series]:
        if not ct_root.is_dir():
            return
        for patient in sorted(ct_root.iterdir(), key=lambda p: _patient_key(p.name)):
            if not patient.is_dir():
                continue
            images = sorted((patient / "DICOM_anon").glob("*.dcm"), key=_slice_index)
            labels = sorted((patient / "Ground").glob("*.png"), key=_slice_index)
            if not images or not labels:
                continue
            images, labels = pair_slices(images, labels)
            yield Series(
                dataset="chaos", modality=MODALITY["CT"], subject=f"CT-{patient.name}",
                images=images, labels=labels, targets=dict(CT_TARGETS),
            )

    def _mr_series(self, mr_root: Path) -> Iterator[Series]:
        if not mr_root.is_dir():
            return
        for patient in sorted(mr_root.iterdir(), key=lambda p: _patient_key(p.name)):
            if not patient.is_dir():
                continue
            for sequence in ("T1DUAL", "T2SPIR"):
                folder = patient / sequence
                if not folder.is_dir():
                    continue
                dicom_root = folder / "DICOM_anon"
                # InPhase only; see the module docstring for why OutPhase is dropped.
                search = dicom_root / "InPhase" if (dicom_root / "InPhase").is_dir() else dicom_root
                images = sorted(search.glob("*.dcm"), key=_slice_index)
                labels = sorted((folder / "Ground").glob("*.png"), key=_slice_index)
                if not images or not labels:
                    continue
                images, labels = pair_slices(images, labels)
                yield Series(
                    dataset="chaos", modality=MODALITY[sequence],
                    subject=f"MR-{patient.name}-{sequence}",
                    images=images, labels=labels, targets=dict(MR_TARGETS),
                )


def _patient_key(name: str) -> tuple:
    """Numeric where possible, so patient 2 sorts before patient 10."""
    return (0, int(name)) if name.isdigit() else (1, name)


register("chaos")(ChaosAdapter)
