"""Stage 3: reduce candidate-level predictions to reportable tables.

    python -m samed.cli.analyse --results results/raw/sam_vit_b --out results/tables

Writes two CSVs and prints them:

* ``per_target.csv`` - DICE per object-modality target and strategy, under the
  paper's oracle rule and under a deployable one.
* ``per_strategy.csv`` - the same aggregated over targets, which is where the
  oracle gap's dependence on prompt ambiguity shows up.

Every figure in the report is regenerated from these, so the raw per-candidate
shards never need to be shipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from samed.analysis import load_results, select_per_prompt, summarise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, type=Path,
                        help="a shard CSV, or a directory of them")
    parser.add_argument("--out", type=Path, help="directory for the summary CSVs")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    return parser


def _print(title: str, frame, columns: dict[str, str]) -> None:
    print(f"\n{title}")
    header = "".join(f"{label:>{width}}" for label, width in columns.values())
    print(header)
    print("-" * len(header))
    for _, row in frame.iterrows():
        cells = []
        for column, (_, width) in zip(columns, columns.values()):
            value = row[column]
            cells.append(f"{value:>{width}.3f}" if isinstance(value, float)
                         else f"{str(value):>{width}}")
        print("".join(cells))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    selected = select_per_prompt(load_results(args.results))
    per_target = summarise(selected, by=("modality", "target", "strategy"), seed=args.seed)
    per_strategy = summarise(selected, by=("strategy",), seed=args.seed)

    _print("DICE per object-modality target", per_target, {
        "modality": ("modality", 10), "target": ("target", 14),
        "strategy": ("strat", 7), "n": ("n", 6),
        "dice_oracle": ("oracle", 9), "dice_score": ("deployable", 12),
        "oracle_gap": ("gap", 8),
    })

    _print("Aggregated over targets", per_strategy, {
        "strategy": ("strategy", 10), "n": ("n", 7), "n_clusters": ("patients", 10),
        "dice_oracle": ("oracle", 9), "dice_score": ("deployable", 12),
        "oracle_gap": ("gap", 8), "oracle_gap_lo": ("gap 95% lo", 12),
        "oracle_gap_hi": ("hi", 8),
    })

    print(
        "\nIntervals are cluster bootstraps over subjects: slices of one scan are\n"
        "near-copies, so `n` masks carry far less independent evidence than `n`\n"
        "suggests - `patients` is the count that matters.\n"
        "\n`oracle` is the paper's rule: keep the candidate that scores best against\n"
        "the ground truth. It needs the answer at inference time, so it is an upper\n"
        "bound. `deployable` uses the model's own quality head. The gap is what the\n"
        "published numbers overstate - and it shrinks as prompts get less ambiguous,\n"
        "so it compresses the very differences between strategies that the paper's\n"
        "conclusions rest on."
    )

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        per_target.to_csv(args.out / "per_target.csv", index=False)
        per_strategy.to_csv(args.out / "per_strategy.csv", index=False)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
