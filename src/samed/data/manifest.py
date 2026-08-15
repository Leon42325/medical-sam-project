"""The manifest: one row per annotated object instance, and the single source
of truth for what gets evaluated.

Every stage reads the same file, so the evaluation set is defined once, is
reviewable as a diff, and is reproducible from the repository alone. Sharding is
positional, so a shard's contents depend only on the manifest and the shard
count - never on filesystem order or on which node picked up the job.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

from samed.data.sampling import MaskRecord

__all__ = ["ManifestRow", "read_manifest", "write_manifest", "shard_of"]


@dataclass(frozen=True)
class ManifestRow:
    """One object instance: where its image and label live, and how to cut it out.

    ``label_value`` is the category code inside the label PNG (the paper resets
    label pixel values per object category, Sec. 2.2), so one label file can
    carry several targets.
    """

    dataset: str
    modality: str
    target: str
    subject: str
    image_id: str
    image_path: str
    label_path: str
    label_value: int
    slice_index: int | None = None
    patient: str = ""

    def as_record(self) -> MaskRecord:
        return MaskRecord(
            dataset=self.dataset,
            modality=self.modality,
            target=self.target,
            subject=self.subject,
            image_id=self.image_id,
            slice_index=self.slice_index,
            patient=self.patient,
        )

    @property
    def key(self) -> str:
        """Stable identifier for this instance's outputs on disk."""
        return f"{self.dataset}__{self.modality}__{self.target}__{self.image_id}"


_FIELDS = [f.name for f in fields(ManifestRow)]


def write_manifest(path: str | Path, rows: Iterable[ManifestRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_manifest(path: str | Path) -> list[ManifestRow]:
    with Path(path).open(newline="") as handle:
        return [
            ManifestRow(
                dataset=r["dataset"],
                modality=r["modality"],
                target=r["target"],
                subject=r["subject"],
                image_id=r["image_id"],
                image_path=r["image_path"],
                label_path=r["label_path"],
                label_value=int(r["label_value"]),
                slice_index=int(r["slice_index"]) if r.get("slice_index") else None,
                patient=r.get("patient") or "",
            )
            for r in csv.DictReader(handle)
        ]


T = TypeVar("T")


def shard_of(items: Sequence[T], shard: int, num_shards: int) -> list[T]:
    """The ``shard``-th of ``num_shards`` interleaved slices of ``items``.

    Interleaved rather than contiguous so that every shard sees a mix of datasets
    and image sizes; contiguous blocks would put all of one large dataset into
    one task and blow its walltime while the others idle.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} out of range for {num_shards} shards")
    return list(items[shard::num_shards])
