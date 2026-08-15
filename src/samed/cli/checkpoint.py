"""Verify a model checkpoint before any result is attributed to it.

    python -m samed.cli.checkpoint --checkpoint medsam_vit_b.pth --variant vit_b \
        --reference sam_vit_b_01ec64.pth

Model weights get the same treatment as datasets (see the provenance policy in
``configs/datasets.yaml``), and for the same reason: a wrong file produces
plausible numbers under the wrong name, and nothing raises an error.

Checksums are the usual answer and are not available here - MedSAM, SAM-Med2D
and MedSAM2 publish no official digest, and their weights circulate through
Google Drive, Kaggle and Hugging Face re-uploads. Three checks that do not need
one:

1. **Structure.** The state dict must load into the architecture it claims to
   be, with no missing or unexpected keys. A converted or truncated file fails
   here.
2. **Divergence.** A medical fine-tune must actually differ from the SAM
   checkpoint it started from. The failure this catches is a mirror that is
   vanilla SAM under a medical name - which would quietly turn "medical
   fine-tuning gives no benefit" into a finding about a filename.
3. **Which parts moved.** MedSAM fine-tuned the mask decoder while freezing the
   image encoder, so an authentic checkpoint diverges in the decoder far more
   than in the encoder. A file that diverges uniformly is a different model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["describe", "compare", "summarise_sections"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--variant", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--reference", type=Path,
                        help="the SAM checkpoint this one was fine-tuned from")
    return parser


def describe(state_dict: dict) -> dict:
    """Tensor count, parameter count and per-section breakdown."""
    sections: dict[str, int] = {}
    total = 0
    for key, tensor in state_dict.items():
        count = int(getattr(tensor, "numel", lambda: 0)())
        total += count
        sections[key.split(".")[0]] = sections.get(key.split(".")[0], 0) + count
    return {"tensors": len(state_dict), "parameters": total, "sections": sections}


def compare(candidate: dict, reference: dict) -> dict:
    """How far a checkpoint has moved from the one it was fine-tuned from."""
    shared = sorted(set(candidate) & set(reference))
    if not shared:
        return {"shared": 0, "identical": 0, "sections": {}}

    identical = 0
    drift: dict[str, list[float]] = {}
    for key in shared:
        a, b = candidate[key], reference[key]
        if getattr(a, "shape", None) != getattr(b, "shape", None):
            continue
        difference = float((a.double() - b.double()).abs().mean())
        if difference == 0.0:
            identical += 1
        drift.setdefault(key.split(".")[0], []).append(difference)

    return {
        "shared": len(shared),
        "identical": identical,
        "sections": {
            section: sum(values) / len(values) for section, values in sorted(drift.items())
        },
    }


def summarise_sections(sections: dict[str, float]) -> str:
    return "\n".join(f"    {name:<24}{value:.3e}" for name, value in sections.items())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch
    from segment_anything import sam_model_registry

    print(f"checkpoint: {args.checkpoint}")
    print(f"  size: {args.checkpoint.stat().st_size / 1e9:.2f} GB")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state) if isinstance(state, dict) else state

    facts = describe(state)
    print(f"  tensors: {facts['tensors']}, parameters: {facts['parameters'] / 1e6:.1f}M")
    for section, count in sorted(facts["sections"].items()):
        print(f"    {section:<24}{count / 1e6:>8.1f}M")

    print(f"\nloading into the {args.variant} architecture:")
    model = sam_model_registry[args.variant]()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing or unexpected:
        for key in (missing + unexpected)[:5]:
            print(f"    {key}")
        print("  REJECTED: this is not a plain "
              f"{args.variant} checkpoint of the architecture it claims to be")
        return 1
    print("  ok - structure matches")

    if args.reference:
        print(f"\ncompared with {args.reference.name}:")
        base = torch.load(args.reference, map_location="cpu", weights_only=True)
        base = base.get("model", base) if isinstance(base, dict) else base
        result = compare(state, base)
        print(f"  shared tensors: {result['shared']}, "
              f"bit-identical: {result['identical']}")
        print("  mean absolute difference by section:")
        print(summarise_sections(result["sections"]))

        if result["identical"] == result["shared"]:
            print("\n  REJECTED: identical to the reference - this is vanilla SAM "
                  "under another name, not a fine-tune")
            return 1
        decoder = result["sections"].get("mask_decoder", 0.0)
        encoder = result["sections"].get("image_encoder", 0.0)
        print(f"\n  decoder moved {decoder:.3e}, encoder moved {encoder:.3e}")
        if encoder > 0 and decoder / encoder < 2:
            print("  NOTE: the encoder moved about as much as the decoder. MedSAM "
                  "froze its encoder, so verify which model this file really is.")

    print("\naccepted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
