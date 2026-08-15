"""Stage 2b: the automatic *everything* mode, strategy S1.

    python -m samed.cli.everything --model sam_vit_b --checkpoint ... \
        --manifest data/manifest.csv --images data/prepared/images \
        --labels data/prepared/labels --out results/raw/... --points-per-side 32

S1 is the only strategy with no prompt: SAM samples a grid over the whole image,
segments everything it finds, and the evaluation asks which of those masks
corresponds to the target. It is also the strategy where the paper reports the
worst performance (Sec. 4.4), and where its mask-matching rule does the most
work - so it is the sharpest test of the oracle-gap finding.

Why it needs its own stage:

* **No embedding reuse.** ``SamAutomaticMaskGenerator`` crops and tiles the
  image itself, so the cache that makes the prompted stages nearly free does
  not apply. This stage is ~2 s per image for ViT-B and ~3 s for ViT-H against
  ~0.01 s for prompt decoding - two orders of magnitude more expensive.
* **Many candidates, not three.** A single image yields tens to hundreds of
  masks. Every rule is still reproduced exactly, but only the masks a rule
  would actually pick are scored and stored; see
  :func:`samed.scoring.candidates_any_rule_would_pick`.
* **One generation serves every target.** The masks do not depend on which
  organ is being looked for, so a slice is segmented once and matched against
  each of its targets in turn.

``--points-per-side`` is the parameter of the paper's Table 5 ablation, which
runs it from 8 to 256; 32 is SAM's default and the value used everywhere else.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from samed.data.manifest import read_manifest, shard_of
from samed.models import create
from samed.scoring import FIELDS, candidates_any_rule_would_pick, score_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    grid = args.points_per_side
    destination = args.out / (
        f"shard-{args.shard:04d}-of-{args.num_shards:04d}"
        + (f"-grid{grid}" if grid != 32 else "")
        + ".csv"
    )
    if args.skip_existing and destination.exists():
        print(f"shard {args.shard}: already done")
        return 0

    from samed.cli.embed import load_image
    from samed.cli.predict import load_mask

    rows = shard_of(read_manifest(args.manifest), args.shard, args.num_shards)
    model = create(args.model, checkpoint=args.checkpoint, device=args.device)
    if not model.supports_strategy("S1"):
        print(f"{args.model} has no automatic mode; nothing to do")
        return 0

    # One generation per image, shared by every target annotated on that slice.
    by_image: dict[str, list] = defaultdict(list)
    for row in rows:
        by_image[row.image_id].append(row)

    temporary = destination.with_suffix(".csv.partial")
    written = generated = 0
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        for position, (image_id, image_rows) in enumerate(sorted(by_image.items())):
            image = load_image(args.images / image_rows[0].image_path)
            masks = model.everything(image, points_per_side=grid)
            generated += len(masks)

            for row in image_rows:
                ground_truth = load_mask(args.labels / row.label_path, row.label_value)
                if not ground_truth.any() or len(masks) == 0:
                    continue
                writer.writerows(score_candidates(
                    masks, ground_truth, row, model=args.model, strategy="S1",
                    indices=candidates_any_rule_would_pick(masks, ground_truth),
                ))
                written += 1

            if position % 25 == 0:
                print(f"  {position + 1}/{len(by_image)} images, "
                      f"{generated / max(position + 1, 1):.0f} masks each on average",
                      flush=True)

    temporary.replace(destination)
    print(f"shard {args.shard}/{args.num_shards}: {len(by_image)} images, "
          f"{generated} masks generated, {written} target matches -> {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
