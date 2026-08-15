"""Stage 0b: turn a downloaded dataset into the uniform form the study assumes.

    python -m samed.cli.prepare --dataset chaos \
        --root $WORK/medical-sam-project/data/chaos \
        --out  $WORK/medical-sam-project/data/prepared \
        --max-per-target 300 --save-overlays 8

Applies the paper's preprocessing (Sec. 2.2) and then our sampling protocol,
producing 8-bit PNG slices, their label maps, and one manifest row per annotated
object instance. Every later stage reads only the manifest.

The preprocessing lives in :mod:`samed.data.preprocess` and is applied
identically to every dataset; the adapter only says where the files are and what
the label values mean. ``--save-overlays`` writes a sample of image/label
overlays, which is the practical way to confirm that slices and annotations were
paired correctly - a silent misalignment would invalidate every number
downstream while breaking nothing visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from samed.data.adapters import Series, available, create
from samed.data.manifest import ManifestRow, write_manifest
from samed.data.preprocess import MIN_LABEL_AREA, min_max_normalise, read_pixels
from samed.data.sampling import stratified_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True, choices=available())
    parser.add_argument("--root", required=True, type=Path, help="the download")
    parser.add_argument("--out", required=True, type=Path, help="prepared data root")
    parser.add_argument("--max-per-target", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-label-area", type=int, default=MIN_LABEL_AREA)
    parser.add_argument(
        "--normalise-scope", choices=["volume", "slice"], default="volume",
        help="min-max over the whole series (default) or per slice; see Sec. 2.2",
    )
    parser.add_argument(
        "--save-overlays", type=int, default=0,
        help="write N image/label overlays per series for visual QC of the pairing",
    )
    return parser


def _write_png(path: Path, array: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), array)


def _overlay(image: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Grey slice with its annotation tinted, for eyeballing the pairing."""
    import cv2

    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for value in np.unique(label):
        if value == 0:
            continue
        # int() because label maps are uint8 and these products overflow in
        # that dtype. Modulo 251 (prime, and coprime to the usual 255-valued
        # binary masks) rather than 255: under mod 255 a label of 255 maps to
        # pure black on every channel, which tinted CHAOS CT annotations
        # invisible - the exact case the overlays exist to inspect. The +40
        # floor keeps every category clearly coloured against grey tissue.
        code = int(value)
        colour = np.array([40 + (code * 61) % 211,
                           40 + (code * 137) % 211,
                           40 + (code * 211) % 211])
        canvas[label == value] = (0.55 * canvas[label == value] + 0.45 * colour).astype(np.uint8)
    return canvas


def prepare_series(
    series: Series,
    out: Path,
    *,
    min_label_area: int,
    normalise_scope: str,
    save_overlays: int,
) -> list[ManifestRow]:
    """Normalise one series, export the slices worth keeping, and describe them."""
    import cv2

    volume = np.stack([read_pixels(path) for path in series.images])
    normalised = min_max_normalise(volume, scope=normalise_scope, axis=0)

    rows: list[ManifestRow] = []
    written: set[int] = set()
    overlays_left = save_overlays

    for index, label_path in enumerate(series.labels):
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError(f"could not read label {label_path}")
        if label.ndim == 3:
            label = label[..., 0]
        if label.shape != normalised[index].shape:
            raise ValueError(
                f"{series.key} slice {index}: image is {normalised[index].shape} but "
                f"its annotation {label_path.name} is {label.shape}"
            )

        present = [
            (target, value) for target, value in series.targets.items()
            if int((label == value).sum()) > min_label_area
        ]
        if not present:
            continue

        image_id = f"{series.key}_{index:04d}"
        image_rel = f"{series.dataset}/{image_id}.png"
        label_rel = f"{series.dataset}/{image_id}_label.png"

        if index not in written:
            _write_png(out / "images" / image_rel, normalised[index])
            _write_png(out / "labels" / label_rel, label)
            written.add(index)
            if overlays_left > 0:
                _write_png(out / "overlays" / f"{image_id}.png",
                           _overlay(normalised[index], label))
                overlays_left -= 1

        for target, value in present:
            rows.append(ManifestRow(
                dataset=series.dataset, modality=series.modality, target=target,
                subject=series.subject, image_id=image_id,
                image_path=image_rel, label_path=label_rel,
                label_value=value, slice_index=index,
            ))

    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adapter = create(args.dataset)

    all_rows: list[ManifestRow] = []
    for series in adapter.series(args.root):
        rows = prepare_series(
            series, args.out,
            min_label_area=args.min_label_area,
            normalise_scope=args.normalise_scope,
            save_overlays=args.save_overlays,
        )
        all_rows.extend(rows)
        print(f"  {series.key:<28} {len(series.images):>4} slices -> {len(rows):>4} instances")

    if not all_rows:
        print("nothing to prepare - no slice met the label-area threshold")
        return 1

    records = {row.image_id + row.target: row for row in all_rows}
    sampled = stratified_sample(
        (row.as_record() for row in all_rows),
        max_per_target=args.max_per_target, seed=args.seed,
    )
    selected = [records[record.image_id + record.target] for record in sampled]

    manifest = args.out / f"manifest-{args.dataset}.csv"
    write_manifest(manifest, selected)

    targets = sorted({(r.modality, r.target) for r in selected})
    print(f"\n{len(all_rows)} instances found, {len(selected)} sampled "
          f"across {len(targets)} object-modality targets:")
    for modality, target in targets:
        count = sum(1 for r in selected if r.modality == modality and r.target == target)
        print(f"  {modality:<10} {target:<15} {count:>4}")
    print(f"\nmanifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
