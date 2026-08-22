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

    def __init__(self, checkpoint: str = "", device: str = "cpu", **kwargs) -> None:
        self.checkpoint = checkpoint
        self.name = kwargs.get("name") or self.name
        self.lora = kwargs.get("lora")

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
    rows = _read_rows(out / "prompted-shard-0000-of-0001.csv")

    assert len(rows) == 4 * 3, "4 objects x 3 candidate masks"
    assert {r["strategy"] for r in rows} == {"S5"}
    assert {int(r["candidate"]) for r in rows} == {0, 1, 2}
    assert all(float(r["gt_area"]) == 400 for r in rows)


def test_predict_covers_every_requested_strategy(workspace):
    embeddings = _run_embed(workspace)
    out = _run_predict(workspace, embeddings, strategies="S2,S3,S4,S5,S6")
    rows = _read_rows(out / "prompted-shard-0000-of-0001.csv")
    assert {r["strategy"] for r in rows} == {"S2", "S3", "S4", "S5", "S6"}


def test_selection_rules_are_recoverable_from_the_table(workspace):
    """The reason masks are not stored: the oracle gap is a groupby on this file."""
    embeddings = _run_embed(workspace)
    out = _run_predict(workspace, embeddings, strategies="S5")
    rows = _read_rows(out / "prompted-shard-0000-of-0001.csv")

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
        workspace, embeddings, strategies="S5", out=workspace / "r-clean") / "prompted-shard-0000-of-0001.csv")
    shaken = _read_rows(_run_predict(
        workspace, embeddings, strategies="S5", jitter="20-30", seed=1,
        out=workspace / "r-jitter") / "prompted-shard-0000-of-0001.csv")
    again = _read_rows(_run_predict(
        workspace, embeddings, strategies="S5", jitter="20-30", seed=1,
        out=workspace / "r-jitter2") / "prompted-shard-0000-of-0001.csv")

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
        seen += [r["image_id"] for r in _read_rows(out / f"prompted-shard-{shard:04d}-of-0003.csv")]

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


def test_shard_outputs_name_the_split_they_came_from(workspace):
    """A 16-way run must not silently reuse an 8-way run's shard 0.

    The two cover different rows, so --skip-existing accepting the stale file
    leaves the combined table double-counting whatever both contained.
    """
    embeddings = _run_embed(workspace)
    out = workspace / "named"
    _run_predict(workspace, embeddings, strategies="S5", out=out, shard=0, num_shards=2)
    _run_predict(workspace, embeddings, strategies="S5", out=out, shard=0, num_shards=4)

    assert sorted(p.name for p in out.glob("*shard-*.csv")) == [
        "prompted-shard-0000-of-0002.csv", "prompted-shard-0000-of-0004.csv"
    ]


# --------------------------------------------------------------------------- #
# Everything mode (S1)
# --------------------------------------------------------------------------- #


class _StubEverything(_StubModel):
    """Emits many masks per image, as the automatic generator does."""

    name = "stub_everything"
    supports = frozenset({"points", "box", "everything"})

    def everything(self, image, points_per_side: int = 32):
        height, width = image.shape[:2]
        candidates, scores = [], []
        for index in range(12):
            canvas = np.zeros((height, width), dtype=bool)
            size = 4 + index * 3
            canvas[8 : 8 + size, 10 : 10 + size] = True
            candidates.append(canvas)
            # The quality head prefers the largest, which is not the best match.
            scores.append(0.1 + index / 12)
        return MaskSet(np.stack(candidates), np.array(scores, dtype=np.float32),
                       self.name, "S1")


register("stub_everything")(_StubEverything)


def _run_everything(workspace: Path, **overrides) -> Path:
    from samed.cli import everything as everything_cli

    out = overrides.pop("out", workspace / "s1")
    argv = [
        "--model", "stub_everything", "--checkpoint", "none",
        "--manifest", str(workspace / "manifest.csv"),
        "--images", str(workspace / "images"), "--labels", str(workspace / "labels"),
        "--out", str(out), "--device", "cpu",
    ]
    argv += _flags(overrides)
    assert everything_cli.main(argv) == 0
    return out


def test_everything_keeps_only_the_masks_a_rule_would_pick(workspace):
    """Storing all of them would make Hausdorff dominate the run for no gain."""
    out = _run_everything(workspace)
    rows = _read_rows(out / "everything-shard-0000-of-0001.csv")

    assert {r["strategy"] for r in rows} == {"S1"}
    assert all(int(r["n_candidates"]) == 12 for r in rows)
    per_object = len(rows) / len({r["image_id"] for r in rows})
    assert per_object <= 4, "at most one row per selection rule"


def test_everything_preserves_both_selection_rules(workspace):
    """The stored subset must reproduce oracle and deployable exactly."""
    out = _run_everything(workspace)
    rows = _read_rows(out / "everything-shard-0000-of-0001.csv")

    for image_id in {r["image_id"] for r in rows}:
        group = [r for r in rows if r["image_id"] == image_id]
        oracle = max(float(r["dice"]) for r in group)
        deployed = float(max(group, key=lambda r: float(r["predicted_iou"]))["dice"])
        assert oracle >= deployed
        assert any(float(r["candidate"]) == 0 for r in group), "the naive rule needs index 0"


def test_everything_shares_the_schema_with_the_prompted_stages(workspace):
    embeddings = _run_embed(workspace)
    prompted = _read_rows(_run_predict(
        workspace, embeddings, strategies="S5") / "prompted-shard-0000-of-0001.csv")
    automatic = _read_rows(_run_everything(workspace) / "everything-shard-0000-of-0001.csv")
    assert prompted[0].keys() == automatic[0].keys()


def test_everything_records_a_non_default_grid_in_the_filename(workspace):
    """Table 5 sweeps the grid; two sweeps must not overwrite each other."""
    out = _run_everything(workspace, points_per_side=64)
    assert (out / "everything-shard-0000-of-0001-grid64.csv").exists()


def test_the_two_prediction_stages_cannot_overwrite_each_other(workspace):
    """They write into one directory; identical names made S1 skip itself."""
    embeddings = _run_embed(workspace)
    shared = workspace / "both"
    _run_predict(workspace, embeddings, strategies="S5", out=shared,
                 shard=0, num_shards=16, skip_existing=True)
    _run_everything(workspace, out=shared, shard=0, num_shards=16, skip_existing=True)

    names = sorted(p.name for p in shared.glob("*shard-*.csv"))
    assert names == [
        "everything-shard-0000-of-0016.csv", "prompted-shard-0000-of-0016.csv"
    ]

    from samed.analysis import load_results
    assert set(load_results(shared)["strategy"]) == {"S1", "S5"}


# --------------------------------------------------------------------------- #
# Object attributes
# --------------------------------------------------------------------------- #


def _run_attributes(workspace: Path, **overrides) -> Path:
    from samed.cli import attributes as attributes_cli

    out = overrides.pop("out", workspace / "attrs")
    argv = [
        "--manifest", str(workspace / "manifest.csv"),
        "--images", str(workspace / "images"), "--labels", str(workspace / "labels"),
        "--out", str(out),
    ]
    argv += _flags(overrides)
    assert attributes_cli.main(argv) == 0
    return out


def test_attributes_measure_every_object(workspace):
    out = _run_attributes(workspace)
    rows = _read_rows(out / "attributes-shard-0000-of-0001.csv")

    assert len(rows) == 4
    assert all(int(r["area"]) == 400 for r in rows), "each fixture square is 20x20"
    assert all(float(r["aspect_ratio"]) == 1.0 for r in rows)
    # The fixture squares are 200 on a background of 40.
    assert all(abs(float(r["intensity_difference"]) - 160) < 1 for r in rows)


def test_attributes_report_both_fourier_variants(workspace):
    """The paper's criterion and a corrected one, so the difference is measurable
    on real data rather than only on synthetic stars."""
    rows = _read_rows(_run_attributes(workspace) / "attributes-shard-0000-of-0001.csv")

    for row in rows:
        assert row["fourier_stop_paper"] in {"dice_target", "no_improvement", "max_order"}
        assert row["fourier_stop_corrected"] in {"dice_target", "no_improvement", "max_order"}
        # A square is simple, so both variants should fit it quickly.
        assert float(row["fourier_paper"]) < 50
        assert int(row["fourier_order_corrected"]) >= 1


def test_attribute_shards_partition_the_manifest(workspace):
    out = workspace / "attrs-sharded"
    seen = []
    for shard in range(2):
        _run_attributes(workspace, out=out, shard=shard, num_shards=2)
        seen += [r["image_id"] for r in
                 _read_rows(out / f"attributes-shard-{shard:04d}-of-0002.csv")]
    assert sorted(seen) == [f"img{i}" for i in range(4)]


def test_attributes_join_onto_results(workspace):
    """The two stages must agree on the keys, or Table 6 silently loses rows."""
    from samed.analysis import load_attributes, load_results, merge_attributes, select_per_prompt

    embeddings = _run_embed(workspace)
    results = _run_predict(workspace, embeddings, strategies="S5")
    attributes = _run_attributes(workspace)

    merged = merge_attributes(select_per_prompt(load_results(results)),
                              load_attributes(attributes))
    assert len(merged) == 4
    assert {"area", "fourier_paper", "oracle_gap"} <= set(merged.columns)


def test_a_fine_tuned_arm_is_recorded_under_its_own_name(workspace):
    """Arms and the checkpoint they started from share every other key, so
    without a distinct name their rows collide and load_results rejects the lot."""
    embeddings = _run_embed(workspace)
    out = workspace / "named-arm"
    _run_predict(workspace, embeddings, strategies="S5", out=out,
                 model_name="sam_vit_b_lora_encoder")

    rows = _read_rows(out / "prompted-shard-0000-of-0001.csv")
    assert {r["model"] for r in rows} == {"sam_vit_b_lora_encoder"}


def test_zero_shot_and_fine_tuned_results_coexist(workspace):
    from samed.analysis import load_results

    embeddings = _run_embed(workspace)
    shared = workspace / "both-models"
    _run_predict(workspace, embeddings, strategies="S5", out=shared)
    _run_predict(workspace, embeddings, strategies="S5", out=shared,
                 model_name="sam_vit_b_decoder", num_shards=2)

    assert set(load_results(shared)["model"]) == {"stub_cli_model", "sam_vit_b_decoder"}


def test_a_cache_from_a_different_encoder_is_refused(workspace):
    """The failure that made a LoRA arm score identically to zero-shot.

    The prediction path never runs the image encoder - it reads cached
    embeddings - so an adapted encoder has no effect at all unless its own cache
    is used, and the job reports success either way.
    """
    embeddings = _run_embed(workspace)
    for path in embeddings.glob("*.npz"):
        cached = dict(np.load(path))
        cached["encoder"] = np.array("stub_cli_model+lora:somewhere")
        with path.open("wb") as handle:
            np.savez(handle, **cached)

    with pytest.raises(ValueError, match="produced by .*but this run needs"):
        _run_predict(workspace, embeddings, strategies="S5",
                     out=workspace / "mismatched")


def test_the_cache_records_which_encoder_made_it(workspace):
    embeddings = _run_embed(workspace)
    cached = dict(np.load(next(embeddings.glob("*.npz"))))
    assert str(cached["encoder"]) == "stub_cli_model"


def test_a_legacy_cache_without_the_field_still_loads(workspace):
    """Caches written before the field existed were all made by the base
    checkpoint, so their absence is unambiguous."""
    embeddings = _run_embed(workspace)
    for path in embeddings.glob("*.npz"):
        cached = {k: v for k, v in np.load(path).items() if k != "encoder"}
        with path.open("wb") as handle:
            np.savez(handle, **cached)

    out = _run_predict(workspace, embeddings, strategies="S5", out=workspace / "legacy")
    assert _read_rows(out / "prompted-shard-0000-of-0001.csv")


def test_encoder_identity_separates_adapted_from_base():
    from samed.models.base import encoder_identity

    assert encoder_identity("sam_vit_b") == "sam_vit_b"
    assert encoder_identity("sam_vit_b", "/runs/lora_encoder") == "sam_vit_b+lora:lora_encoder"
    assert encoder_identity("sam_vit_b") != encoder_identity("sam_vit_b", "/x/lora_encoder")
