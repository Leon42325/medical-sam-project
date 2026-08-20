"""Stage 1b: split the manifest by patient, for the fine-tuning experiment.

    python -m samed.cli.split --manifest data/prepared/manifest-chaos.csv \
        --out data/prepared

Writes one manifest per split beside the original. The split is a file, not a
runtime decision, so it can be inspected, diffed and reused by every later
stage - and so that a rerun cannot quietly land a patient on the other side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from samed.data.manifest import read_manifest, write_manifest
from samed.data.split import DEFAULT_FRACTIONS, coverage_gaps, describe_split, split_by_patient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="defaults to the manifest's directory")
    parser.add_argument("--train", type=float, default=DEFAULT_FRACTIONS["train"])
    parser.add_argument("--val", type=float, default=DEFAULT_FRACTIONS["val"])
    parser.add_argument("--test", type=float, default=DEFAULT_FRACTIONS["test"])
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.out or args.manifest.parent

    splits = split_by_patient(
        read_manifest(args.manifest),
        fractions={"train": args.train, "val": args.val, "test": args.test},
        seed=args.seed,
    )

    print(describe_split(splits))

    gaps = coverage_gaps(splits)
    if gaps:
        # Not a warning: a target missing from a split cannot be trained on or
        # measured, and the per-target comparison this exists for would report
        # its absence as silence.
        print("\nsplit rejected - these object-modality targets are missing:")
        for name, targets in gaps.items():
            for target in targets:
                print(f"  {name}: {' / '.join(target)}")
        print("\nToo few patients in some stratum. Try a different seed, or "
              "fewer splits.")
        return 1

    stem = args.manifest.stem
    for name, rows in splits.items():
        destination = out / f"{stem}-{name}.csv"
        write_manifest(destination, rows)
        print(f"  {destination}")

    overlap = set()
    for name, rows in splits.items():
        for other, others in splits.items():
            if name < other:
                overlap |= ({r.patient for r in rows} & {r.patient for r in others})
    assert not overlap, f"patients in more than one split: {sorted(overlap)}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
