"""End-to-end test of the fine-tuning loop, on a miniature SAM.

Runs the real CLI - manifests, images, labels, embedding cache, optimiser,
validation, checkpointing - with only the model swapped for a two-block
encoder. The forward path it exercises is the real one: encode, prompt with a
box, decode, undo SAM's padding, take the loss at the annotation's own
resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
cv2 = pytest.importorskip("cv2")

from segment_anything.build_sam import _build_sam  # noqa: E402

from samed.cli import finetune as finetune_cli  # noqa: E402
from samed.data.manifest import ManifestRow, write_manifest  # noqa: E402

SIZE = 256


def tiny_sam(checkpoint=None):
    return _build_sam(encoder_embed_dim=48, encoder_depth=2, encoder_num_heads=2,
                      encoder_global_attn_indexes=[], checkpoint=None)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(
        "segment_anything.sam_model_registry",
        {"vit_b": tiny_sam, "vit_l": tiny_sam, "vit_h": tiny_sam},
    )

    images, labels, embeddings = (tmp_path / n for n in ("images", "labels", "embeddings"))
    for directory in (images, labels, embeddings):
        directory.mkdir()

    rows = {"train": [], "val": []}
    for split, patients in (("train", range(4)), ("val", range(4, 6))):
        for patient in patients:
            image = np.full((SIZE, SIZE), 40, np.uint8)
            label = np.zeros((SIZE, SIZE), np.uint8)
            top = 60 + patient * 5
            image[top:top + 80, 70:150] = 200
            label[top:top + 80, 70:150] = 255

            name = f"p{patient}"
            cv2.imwrite(str(images / f"{name}.png"), image)
            cv2.imwrite(str(labels / f"{name}.png"), label)
            np.savez(embeddings / f"{name}.npz",
                     features=np.zeros((1, 256, 64, 64), np.float16),
                     input_size=np.array([1024, 1024]),
                     original_size=np.array([SIZE, SIZE]))

            rows[split].append(ManifestRow(
                dataset="toy", modality="CT", target="block", subject=name,
                patient=name, image_id=name, image_path=f"{name}.png",
                label_path=f"{name}.png", label_value=255, slice_index=0,
            ))

    for split, items in rows.items():
        write_manifest(tmp_path / f"manifest-{split}.csv", items)
    return tmp_path


def _run(workspace: Path, arm: str, **overrides) -> Path:
    out = workspace / arm
    argv = [
        "--arm", arm, "--checkpoint", "none", "--device", "cpu",
        "--train", str(workspace / "manifest-train.csv"),
        "--val", str(workspace / "manifest-val.csv"),
        "--images", str(workspace / "images"), "--labels", str(workspace / "labels"),
        "--embeddings", str(workspace / "embeddings"),
        "--out", str(out), "--epochs", "2",
    ]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    assert finetune_cli.main(argv) == 0
    return out


@pytest.mark.parametrize("arm", ["decoder", "lora_encoder", "lora_encoder_decoder"])
def test_every_arm_trains_and_records_its_history(workspace, arm):
    out = _run(workspace, arm)
    history = json.loads((out / "history.json").read_text())

    assert history["arm"] == arm
    assert history["trainable"] > 0
    assert len(history["epochs"]) == 2
    assert all(np.isfinite(e["loss"]) for e in history["epochs"])
    assert 0.0 <= history["best_val_dice"] <= 1.0


def test_each_arm_saves_only_what_it_changed(workspace):
    """375 MB of frozen weights are not a result; the adapter is."""
    decoder = _run(workspace, "decoder")
    assert (decoder / "mask_decoder.pt").exists()
    assert not (decoder / "lora_encoder").exists()

    lora = _run(workspace, "lora_encoder")
    assert (lora / "lora_encoder").exists()
    assert not (lora / "mask_decoder.pt").exists()

    both = _run(workspace, "lora_encoder_decoder")
    assert (both / "lora_encoder").exists() and (both / "mask_decoder.pt").exists()


def test_the_decoder_arm_refuses_to_run_without_the_cache(workspace, capsys):
    """It trains from cached embeddings by design; a missing cache is a
    configuration error, not something to silently work around."""
    argv = [
        "--arm", "decoder", "--checkpoint", "none", "--device", "cpu",
        "--train", str(workspace / "manifest-train.csv"),
        "--val", str(workspace / "manifest-val.csv"),
        "--images", str(workspace / "images"), "--labels", str(workspace / "labels"),
        "--out", str(workspace / "no-cache"), "--epochs", "1",
    ]
    assert finetune_cli.main(argv) == 2
    assert "pass --embeddings" in capsys.readouterr().out


def test_training_reduces_the_loss(workspace):
    """A loop that runs but does not learn would pass every other test here."""
    out = _run(workspace, "decoder", epochs=6, learning_rate=1e-3)
    losses = [e["loss"] for e in json.loads((out / "history.json").read_text())["epochs"]]
    assert losses[-1] < losses[0], f"loss did not fall: {losses}"


def test_an_optimiser_step_moves_only_the_adapter(workspace):
    """Low-rank means the frozen weights stay frozen. Comparing two freshly
    built models would compare two random initialisations instead, so the check
    has to be made on one model, across one step.
    """
    from segment_anything.utils.transforms import ResizeLongestSide

    from samed.finetune import FineTuneConfig, decode, prepare_model, segmentation_loss

    torch.manual_seed(0)
    sam = tiny_sam()
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    sam = prepare_model(sam, FineTuneConfig(arm="lora_encoder"))

    before = {name: parameter.detach().clone()
              for name, parameter in sam.named_parameters()}

    mask = np.zeros((SIZE, SIZE), bool)
    mask[60:140, 70:150] = True
    box = torch.as_tensor(
        transform.apply_boxes(np.array([[70.0, 60.0, 149.0, 139.0]]), mask.shape),
        dtype=torch.float)
    image = torch.zeros(1, 3, 1024, 1024)

    optimiser = torch.optim.AdamW(
        [p for p in sam.parameters() if p.requires_grad], lr=1e-2)
    logits = decode(sam, sam.image_encoder(image), box, (1024, 1024), mask.shape)
    segmentation_loss(logits, torch.as_tensor(mask)[None, None]).backward()
    optimiser.step()

    moved = {name for name, parameter in sam.named_parameters()
             if not torch.equal(parameter.detach(), before[name])}
    assert moved, "the step changed nothing at all"
    assert all("lora" in name for name in moved), sorted(moved)[:5]
