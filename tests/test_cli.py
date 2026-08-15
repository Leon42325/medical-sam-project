"""End-to-end test of the two pipeline stages, driven by a stub model.

Runs the real CLIs over real files on disk - manifest, PNGs, embedding cache,
result table - so the wiring that the Slurm jobs depend on is verified on a
laptop, without weights or a GPU. Only the model itself is substituted.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from samed.cli import embed as embed_cli
from samed.cli import predict as predict_cli
from samed.data.manifest import ManifestRow, read_manifest, shard_of, write_manifest
from samed.models.base import MaskSet, PromptableSegmenter, register
from samed.prompts import Prompt


class _StubModel(PromptableSegmenter):
    """Returns three nested candidates around the prompt, best one in the middle."""

    name = "stub_cli_model"
    supports = frozenset({"points", "box"})
    input_size = 1024

    def __init__(self, checkpoint: str = "", device: str = "cpu") -> None:
        self.checkpoint = checkpoint

    def encode(self, image):
        return {
            "features": np.zeros((1, 4, 8, 8), dtype=np.float16),
            "original_size": np.asarray(image.shape[:2], dtype=np.int64),
        }

    def predict(self, cached, prompt: Prompt) -> MaskSet:
        height, width = (int(v) for v in cached["original_size"])
        if prompt.box is not None:
            x0, y0, x1, y1 = prompt.box
        else:
            xs, ys = prompt.point_coords[:, 0], prompt.point_coords[:, 1]
            x0, y0, x1, y1 = xs.min() - 6, ys.min() - 6, xs.max() + 6, ys.max() + 6

        candidates = []
        for pad in (12, 0, -4):
            canvas = np.zeros((height, width), dtype=bool)
            a, b = int(max(0, y0 - pad)), int(min(height, y1 + pad + 1))
            c, d = int(max(0, x0 - pad)), int(min(width, x1 + pad + 1))
            canvas[a:b, c:d] = True
            candidates.append(canvas)

        # Deliberately mis-ranked: the model rates the oversized mask highest,
        # which is what creates a measurable oracle gap downstream.
        return MaskSet(
            masks=np.stack(candidates),
            scores=np.array([0.9, 0.4, 0.5], dtype=np.float32),
            model=self.name,
            strategy=prompt.strategy,
        )


register("stub_cli_model")(_StubModel)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Four images, each with one square object, plus a manifest."""
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    rows = []
    for index in range(4):
        image = np.full((64, 64), 40, dtype=np.uint8)
        label = np.zeros((64, 64), dtype=np.uint8)
        top, left = 10 + index * 4, 12
        image[top : top + 20, left : left + 20] = 200
        label[top : top + 20, left : left + 20] = 85

        cv2.imwrite(str(images / f"img{index}.png"), image)
        cv2.imwrite(str(labels / f"img{index}.png"), label)
        rows.append(ManifestRow(
            dataset="stub", modality="CT", target="square", subject=f"p{index}",
            image_id=f"img{index}", image_path=f"img{index}.png",
            label_path=f"img{index}.png", label_value=85, slice_index=index,
        ))

    write_manifest(tmp_path / "manifest.csv", rows)
    return tmp_path


def _flags(overrides: dict) -> list[str]:
    """Turn keyword overrides into argv, treating True as a bare switch."""
    argv: list[str] = []
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        argv.append(flag)
        if value is not True:
            argv.append(str(value))
    return argv


def _run_embed(workspace: Path, **overrides) -> Path:
    out = overrides.pop("out", workspace / "embeddings")
    argv = [
        "--model", "stub_cli_model", "--checkpoint", "none",
        "--manifest", str(workspace / "manifest.csv"),
        "--images", str(workspace / "images"), "--out", str(out),
        "--device", "cpu",
    ]
    argv += _flags(overrides)
    assert embed_cli.main(argv) == 0
    return out


def _run_predict(workspace: Path, embeddings: Path, **overrides) -> Path:
    out = overrides.pop("out", workspace / "results")
    argv = [
        "--model", "stub_cli_model", "--checkpoint", "none",
        "--manifest", str(workspace / "manifest.csv"),
        "--embeddings", str(embeddings), "--labels", str(workspace / "labels"),
        "--out", str(out), "--device", "cpu",
    ]
    argv += _flags(overrides)
    assert predict_cli.main(argv) == 0
    return out


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------- #


def test_manifest_round_trips(workspace):
    rows = read_manifest(workspace / "manifest.csv")
    assert len(rows) == 4
    assert rows[0].label_value == 85 and rows[0].slice_index == 0
    assert rows[0].key == "stub__CT__square__img0"


def test_embed_writes_one_cache_entry_per_image(workspace):
    out = _run_embed(workspace)
    assert sorted(p.name for p in out.glob("*.npz")) == [f"img{i}.npz" for i in range(4)]
    assert not list(out.glob("*.partial")), "no truncated files may survive"


def test_embed_skips_existing_work(workspace):
    out = _run_embed(workspace)
    stamps = {p.name: p.stat().st_mtime_ns for p in out.glob("*.npz")}
    _run_embed(workspace, skip_existing=True)  # flag takes no value
    assert {p.name: p.stat().st_mtime_ns for p in out.glob("*.npz")} == stamps


def test_predict_emits_one_row_per_candidate(workspace):
    embeddings = _run_embed(workspace)
    out = _run_predict(workspace, embeddings, strategies="S5")
    rows = _read_rows(out / "shard-0000.csv")

    assert len(rows) == 4 * 3, "4 objects x 3 candidate masks"
    assert {r["strategy"] for r in rows} == {"S5"}
    assert {int(r["candidate"]) for r in rows} == {0, 1, 2}
    assert all(float(r["gt_area"]) == 400 for r in rows)


def test_predict_covers_every_requested_strategy(workspace):
    embeddings = _run_embed(workspace)
    out = _run_predict(workspace, embeddings, strategies="S2,S3,S4,S5,S6")
    rows = _read_rows(out / "shard-0000.csv")
    assert {r["strategy"] for r in rows} == {"S2", "S3", "S4", "S5", "S6"}


def test_selection_rules_are_recoverable_from_the_table(workspace):
    """The reason masks are not stored: the oracle gap is a groupby on this file."""
    embeddings = _run_embed(workspace)
    out = _run_predict(workspace, embeddings, strategies="S5")
    rows = _read_rows(out / "shard-0000.csv")

    by_object: dict[str, list[dict]] = {}
    for row in rows:
        by_object.setdefault(row["image_id"], []).append(row)

    gaps = []
    for candidates in by_object.values():
        oracle = max(float(c["dice"]) for c in candidates)
        deployed = float(max(candidates, key=lambda c: float(c["predicted_iou"]))["dice"])
        gaps.append(oracle - deployed)

    assert all(gap >= 0 for gap in gaps), "the oracle is an upper bound by construction"
    assert max(gaps) > 0.1, "the stub is built to mis-rank, so a gap must show up"


def test_jitter_changes_the_result_and_is_reproducible(workspace):
    embeddings = _run_embed(workspace)
    clean = _read_rows(_run_predict(
        workspace, embeddings, strategies="S5", out=workspace / "r-clean") / "shard-0000.csv")
    shaken = _read_rows(_run_predict(
        workspace, embeddings, strategies="S5", jitter="20-30", seed=1,
        out=workspace / "r-jitter") / "shard-0000.csv")
    again = _read_rows(_run_predict(
        workspace, embeddings, strategies="S5", jitter="20-30", seed=1,
        out=workspace / "r-jitter2") / "shard-0000.csv")

    assert [r["dice"] for r in shaken] == [r["dice"] for r in again], "same seed, same result"
    assert [r["dice"] for r in shaken] != [r["dice"] for r in clean]
    assert all(r["jitter"] == "20-30" for r in shaken)


def test_shards_partition_the_manifest_exactly(workspace):
    embeddings = _run_embed(workspace)
    out = workspace / "sharded"
    seen: list[str] = []
    for shard in range(3):
        _run_predict(workspace, embeddings, strategies="S5", out=out,
                     shard=shard, num_shards=3)
        seen += [r["image_id"] for r in _read_rows(out / f"shard-{shard:04d}.csv")]

    assert sorted(set(seen)) == [f"img{i}" for i in range(4)]
    assert len(seen) == 4 * 3, "no object may be evaluated twice"


def test_shard_of_is_a_partition():
    items = list(range(23))
    recovered: list[int] = []
    for shard in range(4):
        recovered += shard_of(items, shard, 4)
    assert sorted(recovered) == items

    with pytest.raises(ValueError, match="out of range"):
        shard_of(items, 4, 4)


def test_save_masks_every_stores_a_sample(workspace):
    embeddings = _run_embed(workspace)
    out = _run_predict(workspace, embeddings, strategies="S5", save_masks_every=2)
    saved = sorted(p.name for p in (out / "masks").glob("*.npz"))
    assert saved == ["stub__CT__square__img0__S5.npz", "stub__CT__square__img2__S5.npz"]
