"""Stage 4: fine-tune SAM, and vary where the adaptation is allowed to happen.

    python -m samed.cli.finetune --arm lora_encoder \
        --checkpoint sam_vit_b_01ec64.pth \
        --train data/prepared/manifest-chaos-train.csv \
        --val   data/prepared/manifest-chaos-val.csv \
        --images data/prepared/images --labels data/prepared/labels \
        --out runs/lora_encoder

Three arms, differing only in which parameters receive gradient:

* ``decoder``              - the paper's Sec. 4.12 configuration, replicated.
* ``lora_encoder``         - LoRA on the image encoder, decoder frozen.
* ``lora_encoder_decoder`` - both.

Training uses box prompts throughout, as the paper does, with the prompt encoder
frozen in every arm. The split is by patient (see :mod:`samed.data.split`);
training and validation never share a scan.

The ``decoder`` arm reads cached embeddings and never runs the image encoder,
which is what makes it minutes rather than hours. The LoRA arms cannot: their
encoder changes every step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from samed.data.manifest import read_manifest
from samed.finetune import FineTuneConfig, decode, prepare_model, segmentation_loss, trainable_parameters
from samed.models.sam import as_rgb_uint8
from samed.prompts import bounding_box


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True,
                        choices=["decoder", "lora_encoder", "lora_encoder_decoder"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--embeddings", type=Path,
                        help="required for the decoder arm, unused by the LoRA arms")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-targets", default="qkv,proj")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, help="use only N rows, for a smoke test")
    return parser


class Example:
    """One training item: a box prompt and the mask it should produce."""

    __slots__ = ("row", "mask", "box", "image", "embedding")

    def __init__(self, row, mask, box, image=None, embedding=None):
        self.row, self.mask, self.box = row, mask, box
        self.image, self.embedding = image, embedding


def load_examples(rows, images_root: Path, labels_root: Path,
                  embeddings_root: Path | None, *, need_images: bool) -> list[Example]:
    """Read every row once, up front.

    The evaluation subset is capped at 300 masks per target, so the whole split
    fits in memory comfortably and the training loop never touches the shared
    filesystem - which matters on a cluster where that filesystem is the slowest
    thing in the system.
    """
    from samed.cli.embed import load_image
    from samed.cli.predict import load_mask

    examples = []
    for row in rows:
        mask = load_mask(labels_root / row.label_path, row.label_value)
        if not mask.any():
            continue
        image = load_image(images_root / row.image_path) if need_images else None
        embedding = None
        if embeddings_root is not None:
            embedding = dict(np.load(embeddings_root / f"{row.image_id}.npz"))
        examples.append(Example(row, mask, bounding_box(mask), image, embedding))
    return examples


def _batches(items, size, rng):
    order = rng.permutation(len(items))
    for start in range(0, len(order), size):
        yield [items[i] for i in order[start:start + size]]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    config = FineTuneConfig(
        arm=args.arm, lora_rank=args.lora_rank,
        lora_targets=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
        learning_rate=args.learning_rate, batch_size=args.batch_size,
        epochs=args.epochs, seed=args.seed,
    )
    if config.can_use_cached_embeddings and args.embeddings is None:
        print("the decoder arm trains from cached embeddings; pass --embeddings")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    sam = sam_model_registry[args.variant](checkpoint=args.checkpoint)
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    sam = prepare_model(sam, config).to(args.device)

    print(f"arm {config.arm}: {trainable_parameters(sam) / 1e6:.3f}M trainable parameters, "
          f"cached embeddings {'used' if config.can_use_cached_embeddings else 'not usable'}")

    need_images = not config.can_use_cached_embeddings
    splits = {}
    for name, manifest in (("train", args.train), ("val", args.val)):
        rows = read_manifest(manifest)
        if args.limit:
            rows = rows[:args.limit]
        splits[name] = load_examples(
            rows, args.images, args.labels,
            args.embeddings if config.can_use_cached_embeddings else None,
            need_images=need_images,
        )
        print(f"  {name}: {len(splits[name])} examples from "
              f"{len({e.row.patient for e in splits[name]})} patients")

    def forward(example: Example):
        original = example.mask.shape
        box = transform.apply_boxes(example.box[None, :], original)
        box = torch.as_tensor(box, dtype=torch.float, device=args.device)

        if config.can_use_cached_embeddings:
            embeddings = torch.as_tensor(
                example.embedding["features"], dtype=torch.float, device=args.device)
            input_size = tuple(int(v) for v in example.embedding["input_size"])
        else:
            resized = transform.apply_image(as_rgb_uint8(example.image))
            tensor = torch.as_tensor(resized, device=args.device)
            tensor = tensor.permute(2, 0, 1).contiguous()[None, :, :, :]
            input_size = tuple(tensor.shape[-2:])
            embeddings = sam.image_encoder(sam.preprocess(tensor))

        return decode(sam, embeddings, box, input_size, original)

    optimiser = torch.optim.AdamW(
        [p for p in sam.parameters() if p.requires_grad], lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)

    def evaluate() -> float:
        sam.eval()
        scores = []
        with torch.no_grad():
            for example in splits["val"]:
                logits = forward(example)
                prediction = (logits[0, 0] > 0).cpu().numpy()
                truth = example.mask
                total = prediction.sum() + truth.sum()
                scores.append(1.0 if total == 0 else
                              2.0 * np.logical_and(prediction, truth).sum() / total)
        return float(np.mean(scores))

    history, best = [], -1.0
    for epoch in range(config.epochs):
        sam.train()
        started, losses = time.time(), []
        for batch in _batches(splits["train"], config.batch_size, rng):
            optimiser.zero_grad(set_to_none=True)
            loss = 0.0
            for example in batch:
                logits = forward(example)
                target = torch.as_tensor(
                    example.mask, device=args.device)[None, None, :, :]
                loss = loss + segmentation_loss(logits, target)
            (loss / len(batch)).backward()
            optimiser.step()
            losses.append(float(loss.detach()) / len(batch))

        validation = evaluate()
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "val_dice": validation, "seconds": time.time() - started})
        print(f"  epoch {epoch:>3}  loss {np.mean(losses):.4f}  "
              f"val DICE {validation:.4f}  {time.time() - started:.0f}s", flush=True)

        if validation > best:
            best = validation
            save(sam, config, args.out)

    (args.out / "history.json").write_text(json.dumps(
        {"arm": config.arm, "trainable": trainable_parameters(sam),
         "best_val_dice": best, "epochs": history}, indent=2) + "\n")
    print(f"best validation DICE {best:.4f}; weights and history in {args.out}")
    return 0


def save(sam, config: FineTuneConfig, out: Path) -> None:
    """Store only what this arm changed.

    An adapter is a few hundred kilobytes and a decoder a few megabytes, against
    375 MB for the whole model - and keeping the arms' outputs to what they
    actually learned makes the comparison legible on disk as well as in the
    results.
    """
    import torch

    if config.trains_encoder:
        sam.image_encoder.save_pretrained(out / "lora_encoder")
    if config.trains_decoder:
        torch.save(sam.mask_decoder.state_dict(), out / "mask_decoder.pt")


if __name__ == "__main__":
    sys.exit(main())
