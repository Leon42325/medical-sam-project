"""Stage 2c: measure the object attributes the paper's perception analysis rests on.

    python -m samed.cli.attributes --manifest data/prepared/manifest-chaos.csv \
        --images data/prepared/images --labels data/prepared/labels \
        --out results/attributes

One row per annotated object instance, holding the five factors of Huang et al.
Sec. 4.8 - size, aspect ratio, foreground-background intensity difference,
boundary complexity, and modality - which they correlate with DICE in Table 6.

Boundary complexity is computed **twice**, under the paper's own termination
criterion and under a corrected one. That criterion stops the elliptic Fourier
search at the first plateau, and an EFD fit improves in a staircase, so on
synthetic shapes it halts at order 2 while the contour is only fitted around
order 11-13 (see :func:`samed.attributes.fourier_order`). Reporting both is what
turns that observation from a claim about synthetic stars into a measurement of
how much the paper's headline correlation actually moves.

The corrected variant is capped below the paper's limit: without the plateau
rule the search runs until the contour is fitted, and retinal vessels or
adenocarcinoma boundaries would not be fitted at any tractable order. Structures
that hit the cap are marked ``max_order`` in ``fourier_stop_corrected`` rather
than silently reported as if they had converged.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from samed.attributes import area, aspect_ratio, fourier_order, intensity_difference
from samed.data.manifest import read_manifest, shard_of
from samed.scoring import shard_filename

FIELDS = [
    "dataset", "modality", "target", "subject", "patient", "image_id",
    "slice_index", "label_value",
    "area", "aspect_ratio", "intensity_difference",
    "fourier_paper", "fourier_order_paper", "fourier_dice_paper", "fourier_stop_paper",
    "fourier_corrected", "fourier_order_corrected", "fourier_dice_corrected",
    "fourier_stop_corrected",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--max-order", type=int, default=60,
        help="cap for the corrected Fourier search; contours that reach it are flagged",
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def measure(image, mask, *, max_order: int) -> dict:
    """Every attribute for one object instance, both Fourier variants."""
    paper = fourier_order(mask, patience=1)
    corrected = fourier_order(mask, patience=None, max_order=max_order)
    return {
        "area": area(mask),
        "aspect_ratio": round(aspect_ratio(mask), 6),
        "intensity_difference": round(intensity_difference(image, mask), 4),
        "fourier_paper": round(paper.value, 4),
        "fourier_order_paper": paper.order,
        "fourier_dice_paper": round(paper.dice, 6),
        "fourier_stop_paper": paper.stopped_by,
        "fourier_corrected": round(corrected.value, 4),
        "fourier_order_corrected": corrected.order,
        "fourier_dice_corrected": round(corrected.dice, 6),
        "fourier_stop_corrected": corrected.stopped_by,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    destination = args.out / shard_filename("attributes", args.shard, args.num_shards)
    if args.skip_existing and destination.exists():
        print(f"shard {args.shard}: already done")
        return 0

    from samed.cli.embed import load_image
    from samed.cli.predict import load_mask

    rows = shard_of(read_manifest(args.manifest), args.shard, args.num_shards)

    temporary = destination.with_suffix(".csv.partial")
    written = skipped = 0
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        for position, row in enumerate(rows):
            mask = load_mask(args.labels / row.label_path, row.label_value)
            if not mask.any():
                skipped += 1
                continue
            image = load_image(args.images / row.image_path)

            writer.writerow({
                "dataset": row.dataset, "modality": row.modality, "target": row.target,
                "subject": row.subject, "patient": row.patient, "image_id": row.image_id,
                "slice_index": row.slice_index, "label_value": row.label_value,
                **measure(image, mask, max_order=args.max_order),
            })
            written += 1

            if position % 50 == 0:
                print(f"  {position + 1}/{len(rows)}", flush=True)

    temporary.replace(destination)
    print(f"shard {args.shard}/{args.num_shards}: {written} measured, "
          f"{skipped} empty -> {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
