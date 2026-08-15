"""Stage 1: run an image encoder once per image and cache the result.

    python -m samed.cli.embed --model sam_vit_b --checkpoint ... \
        --manifest data/manifest.csv --images data/images --out embeddings/sam_vit_b

One cache entry per *image*, not per manifest row: a slice usually carries
several annotated targets, and they all share one encoding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from samed.data.manifest import read_manifest, shard_of
from samed.models import create


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path, help="root for image_path")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="resume: leave already-encoded images alone (jobs are preemptible)",
    )
    return parser


def load_image(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"could not read image {path}")
    if image.ndim == 3:  # cv2 gives BGR; SAM expects RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    # Deduplicate to images, keeping manifest order so sharding stays stable.
    seen: dict[str, str] = {}
    for row in rows:
        seen.setdefault(row.image_id, row.image_path)
    images = shard_of(sorted(seen.items()), args.shard, args.num_shards)

    model = create(args.model, checkpoint=args.checkpoint, device=args.device)

    encoded = skipped = 0
    for image_id, image_path in images:
        destination = args.out / f"{image_id}.npz"
        if args.skip_existing and destination.exists():
            skipped += 1
            continue
        cached = model.encode(load_image(args.images / image_path))
        # Write to a temporary name first: a job killed mid-write must not leave
        # a truncated file that --skip-existing would then treat as done.
        temporary = destination.with_suffix(".npz.partial")
        # Write through a handle: np.savez appends ".npz" to any path that does
        # not already end in it, which would defeat the atomic rename below.
        with temporary.open("wb") as handle:
            np.savez(handle, **cached)
        temporary.replace(destination)
        encoded += 1

    print(f"shard {args.shard}/{args.num_shards}: encoded {encoded}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
