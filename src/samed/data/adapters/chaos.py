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

**Slice order comes from the DICOM header, not the filename.** CHAOS CT files
are named ``i0000,0000b.dcm``; nothing guarantees that lexical order matches
acquisition order, and a wrong order silently pairs every slice with the wrong
annotation. ``ImagePositionPatient`` is used where present, ``InstanceNumber``
otherwise.
"""

from __future__ import annotations

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


def _slice_position(path: Path) -> tuple:
    """Sort key placing DICOM slices in anatomical order.

    ``ImagePositionPatient[2]`` is the through-plane coordinate and is the
    correct key; ``InstanceNumber`` is the fallback for series that omit it. The
    filename is the last resort and is only ever a tie-break.
    """
    import pydicom

    header = pydicom.dcmread(str(path), stop_before_pixels=True)
    position = getattr(header, "ImagePositionPatient", None)
    if position is not None and len(position) == 3:
        return (0, float(position[2]), path.name)
    instance = getattr(header, "InstanceNumber", None)
    if instance is not None:
        return (1, int(instance), path.name)
    return (2, 0.0, path.name)


def pair_slices(images: list[Path], labels: list[Path]) -> tuple[list[Path], list[Path]]:
    """Align an ordered DICOM series with its annotations.

    Two conventions occur in CHAOS. MR annotations carry the same stem as their
    DICOM slice, so they can be matched by name - which is exact, and immune to
    any ordering mistake. CT annotations are named ``liver_GT_000.png`` and
    carry no reference to the slice, leaving positional pairing as the only
    option: the *i*-th annotation belongs to the *i*-th slice in anatomical
    order.

    Name matching is attempted first for exactly that reason, and positional
    pairing is used only when it cannot apply.
    """
    by_stem = {path.stem: path for path in labels}
    if all(image.stem in by_stem for image in images) and len(by_stem) == len(labels):
        return images, [by_stem[image.stem] for image in images]

    ordered_labels = sorted(labels, key=lambda p: p.name)
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
            images = sorted((patient / "DICOM_anon").glob("*.dcm"), key=_slice_position)
            labels = sorted((patient / "Ground").glob("*.png"))
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
                images = sorted(search.glob("*.dcm"), key=_slice_position)
                labels = sorted((folder / "Ground").glob("*.png"))
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
