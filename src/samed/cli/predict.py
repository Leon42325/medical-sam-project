"""Stage 2: decode masks for strategies S2-S6 and score every candidate.

    python -m samed.cli.predict --model sam_vit_b --checkpoint ... \
        --manifest data/manifest.csv --embeddings embeddings/sam_vit_b \
        --labels data/labels --strategies S2,S3,S4,S5,S6 --out results/raw/...

What is written, and why not the masks
--------------------------------------
One row per *candidate* mask, holding its overlap and distance metrics plus the
model's own predicted IoU. The masks themselves are discarded.

That is a deliberate trade. SAM returns several candidates per prompt, so the
full study produces on the order of a hundred thousand masks; storing them would
cost hundreds of gigabytes and buy nothing, because every downstream question is
answerable from the per-candidate metrics. In particular the selection rules of
:mod:`samed.selection` reduce to group-wise argmax over this table - the paper's
oracle rule is ``argmax(dice)``, the deployable rule is ``argmax(predicted_iou)``
- so the oracle gap that Sec. 9.1 of the plan is about costs one groupby rather
than a second inference pass.

Use ``--save-masks-every`` to keep a periodic sample for qualitative figures.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from samed.data.manifest import read_manifest, shard_of
from samed.models import create
from samed.prompts import build_prompt, jitter_prompt
from samed.scoring import FIELDS, score_candidates

JITTER_LEVELS: dict[str, tuple[float, float] | None] = {
    "none": None,
    "0-10": (0.0, 10.0),
    "10-20": (10.0, 20.0),
    "20-30": (20.0, 30.0),
}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path, help="root for label_path")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--strategies", default="S2,S3,S4,S5,S6")
    parser.add_argument("--jitter", default="none", choices=sorted(JITTER_LEVELS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--save-masks-every",
        type=int,
        default=0,
        help="also store the masks of every Nth row, for qualitative figures",
    )
    return parser


def load_mask(path: Path, label_value: int) -> np.ndarray:
    import cv2

    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(f"could not read label {path}")
    if label.ndim == 3:
        label = label[..., 0]
    return label == label_value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    # The shard count belongs in the name. Without it a run split 16 ways
    # reuses the output of an earlier 8-way run under --skip-existing, and the
    # two cover different rows: the combined table then double-counts whatever
    # the coarser shard also contained, silently reweighting the aggregate.
    destination = args.out / f"shard-{args.shard:04d}-of-{args.num_shards:04d}.csv"
    if args.skip_existing and destination.exists():
        print(f"shard {args.shard}: already done")
        return 0

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    band = JITTER_LEVELS[args.jitter]
    rows = shard_of(read_manifest(args.manifest), args.shard, args.num_shards)
    model = create(args.model, checkpoint=args.checkpoint, device=args.device)

    applicable = [s for s in strategies if model.supports_strategy(s)]
    if skipped := sorted(set(strategies) - set(applicable)):
        # Not an error: MedSAM is box-only by construction, so S2-S4 and S6 are
        # genuinely N/A for it and must be absent rather than approximated.
        print(f"{args.model} does not support {', '.join(skipped)} - omitted")

    mask_dir = args.out / "masks"
    if args.save_masks_every:
        mask_dir.mkdir(exist_ok=True)

    temporary = destination.with_suffix(".csv.partial")
    written = 0
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        for position, row in enumerate(rows):
            ground_truth = load_mask(args.labels / row.label_path, row.label_value)
            if not ground_truth.any():
                continue
            cached = dict(np.load(args.embeddings / f"{row.image_id}.npz"))

            for strategy in applicable:
                prompt = build_prompt(ground_truth, strategy)
                if band is not None:
                    # Seed per row and strategy so a rerun of one shard
                    # reproduces exactly the same perturbation.
                    rng = np.random.default_rng([args.seed, position, ord(strategy[1])])
                    prompt = jitter_prompt(
                        prompt, *band, rng, image_shape=ground_truth.shape
                    )

                masks = model.predict(cached, prompt)
                writer.writerows(score_candidates(
                    masks, ground_truth, row, model=args.model,
                    strategy=strategy, jitter=args.jitter, seed=args.seed,
                ))
                written += len(masks)

                if args.save_masks_every and position % args.save_masks_every == 0:
                    np.savez_compressed(
                        mask_dir / f"{row.key}__{strategy}.npz",
                        masks=np.packbits(masks.masks, axis=None),
                        shape=np.array(masks.masks.shape),
                        scores=masks.scores,
                    )

    temporary.replace(destination)
    print(f"shard {args.shard}/{args.num_shards}: {written} candidate rows -> {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
