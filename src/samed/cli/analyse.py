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

from samed.analysis import (
    attribute_correlations,
    load_attributes,
    load_results,
    merge_attributes,
    paired_comparison,
    select_per_prompt,
    summarise,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, type=Path,
                        help="a shard CSV, or a directory of them")
    parser.add_argument("--out", type=Path, help="directory for the summary CSVs")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--baseline", help="model to compare the others against, paired per prompt")
    parser.add_argument("--attributes", type=Path,
                        help="object attributes from samed.cli.attributes; enables Table 6")
    parser.add_argument(
        "--jitter", default="none",
        help="which perturbation level the main tables describe: a level, or 'all'. "
             "Defaults to the unperturbed runs, so that submitting the jitter study "
             "cannot silently average perturbed results into the headline numbers.",
    )
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


CORRELATION_COLUMNS = {
    "strategy": ("strat", 7), "n": ("n", 7),
    "area": ("size", 8), "area_sig": ("sig", 5),
    "intensity_difference": ("contrast", 10), "intensity_difference_sig": ("sig", 5),
    "fourier_paper": ("complexity", 12), "fourier_paper_sig": ("sig", 5),
    "modality_code": ("modality", 10),
    "aspect_ratio": ("aspect", 8),
}


def _jitter_table(selected, args) -> None:
    """Table 8: how far performance falls as the prompt is displaced.

    The paper reports a DICE *drop* against the unperturbed run, so that is what
    is shown - together with the level it is measured from, since a drop is only
    interpretable next to its baseline.
    """
    table = summarise(selected, by=("jitter", "strategy"), seed=args.seed)
    baseline = table[table["jitter"] == "none"].set_index("strategy")["dice_score"]
    table["drop"] = table.apply(
        lambda row: baseline.get(row["strategy"], float("nan")) - row["dice_score"], axis=1
    )
    _print("Prompt perturbation (paper Table 8)", table.sort_values(["strategy", "jitter"]), {
        "strategy": ("strat", 7), "jitter": ("shift px", 10),
        "n_clusters": ("patients", 10),
        "dice_score": ("deployable", 12), "drop": ("drop", 8),
        "dice_oracle": ("oracle", 9), "oracle_gap": ("gap", 8),
    })
    print(
        "\n`drop` is measured against the same strategy unperturbed. The paper reports\n"
        "this under its oracle rule; both columns are here, because a rule that picks\n"
        "the best mask by ground truth can absorb a displaced prompt that a deployable\n"
        "one cannot."
    )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out / "jitter.csv", index=False)


def _correlations(merged, args) -> None:
    """Table 6, plus the same analysis applied to the oracle gap."""
    published = attribute_correlations(merged, outcome="dice_score", seed=args.seed)
    _print("Partial rank correlation of DICE with object attributes (paper Table 6)",
           published, CORRELATION_COLUMNS)

    corrected = attribute_correlations(
        merged, outcome="dice_score",
        predictors=("area", "intensity_difference", "fourier_corrected",
                    "modality_code", "aspect_ratio"),
        seed=args.seed,
    )
    _print("The same, with the corrected boundary-complexity measure", corrected, {
        **{k: v for k, v in CORRELATION_COLUMNS.items() if not k.startswith("fourier")},
        "fourier_corrected": ("complexity", 12), "fourier_corrected_sig": ("sig", 5),
    })

    gap = attribute_correlations(merged, outcome="oracle_gap", seed=args.seed)
    _print("Partial rank correlation of the ORACLE GAP with the same attributes",
           gap, CORRELATION_COLUMNS)
    print(
        "\nThe last table is not in the paper. It asks what makes a prompt ambiguous:\n"
        "the gap varies by more than an order of magnitude across targets (liver ~0.42,\n"
        "T2W kidney ~0.03), and these are the object properties that go with it.\n"
        "\nModality is included because the paper includes it, having mapped a nominal\n"
        "variable to arbitrary integers. Any other coding gives a different number, so\n"
        "the column is reported without a significance flag and should not be read."
    )

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        published.to_csv(args.out / "correlations_dice.csv", index=False)
        corrected.to_csv(args.out / "correlations_dice_corrected_fourier.csv", index=False)
        gap.to_csv(args.out / "correlations_oracle_gap.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    results = load_results(args.results)
    levels = sorted(results["jitter"].astype(str).unique())
    if args.jitter != "all":
        if args.jitter not in levels:
            print(f"no results at jitter level {args.jitter!r}; found {levels}")
            return 2
        clean = results[results["jitter"].astype(str) == args.jitter]
    else:
        clean = results
    selected = select_per_prompt(clean)

    if len(levels) > 1 and args.jitter != "all":
        print(f"jitter levels present: {', '.join(levels)}; "
              f"tables below describe {args.jitter!r} only")
        _jitter_table(select_per_prompt(results), args)
    per_target = summarise(selected, by=("modality", "target", "strategy"), seed=args.seed)
    per_strategy = summarise(selected, by=("strategy",), seed=args.seed)

    _print("DICE per object-modality target", per_target, {
        "modality": ("modality", 10), "target": ("target", 14),
        "strategy": ("strat", 7), "n": ("n", 6),
        "dice_oracle": ("oracle", 9), "dice_score": ("deployable", 12),
        "oracle_gap": ("gap", 8),
    })

    if selected["model"].nunique() > 1:
        # The comparison that decides how far the oracle-gap finding reaches. If
        # the gap is a property of prompt ambiguity it holds across model sizes;
        # if it is an artefact of a weak model it shrinks with scale, and the
        # criticism applies only to ViT-B.
        _print("By model", summarise(selected, by=("model", "strategy"), seed=args.seed), {
            "model": ("model", 12), "strategy": ("strat", 7),
            "n_clusters": ("patients", 10),
            "dice_oracle": ("oracle", 9), "dice_score": ("deployable", 12),
            "oracle_gap": ("gap", 8), "oracle_gap_lo": ("gap 95% lo", 12),
            "oracle_gap_hi": ("hi", 8),
        })

    models = sorted(selected["model"].unique())
    baseline = args.baseline or (models[0] if len(models) > 1 else None)
    if baseline and len(models) > 1:
        for model in models:
            if model == baseline:
                continue
            paired = paired_comparison(
                selected, baseline=baseline, other=model, seed=args.seed
            )
            _print(f"{model} minus {baseline}, paired on the same prompts", paired, {
                "strategy": ("strat", 7), "n_prompts": ("prompts", 9),
                "delta_dice_oracle": ("d oracle", 10),
                "delta_dice_oracle_sig": ("sig", 5),
                "delta_dice_score": ("d deployable", 14),
                "delta_dice_score_sig": ("sig", 5),
                "delta_dice_score_lo": ("95% lo", 9),
                "delta_dice_score_hi": ("hi", 9),
            })
            if args.out:
                args.out.mkdir(parents=True, exist_ok=True)
                paired.to_csv(args.out / f"paired_{model}_vs_{baseline}.csv", index=False)

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

    if "S1" in set(selected["strategy"]):
        print(
            "\nS1 is not comparable with the rest. Automatic mode names no target, so\n"
            "its quality head ranks masks by segmentation quality, not by whether they\n"
            "are the organ being looked for. Read its oracle score as `a good mask\n"
            "exists in the output`, and its gap as how much of the published\n"
            "everything-mode performance came from the ground truth supplying the\n"
            "semantics - not as a ranking failure."
        )

    if args.attributes:
        merged = merge_attributes(selected, load_attributes(args.attributes))
        _correlations(merged, args)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        per_target.to_csv(args.out / "per_target.csv", index=False)
        per_strategy.to_csv(args.out / "per_strategy.csv", index=False)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
