"""Tests for the selection rules and the oracle-gap aggregation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from samed.analysis import (
    PROMPT_KEYS,
    bootstrap_ci,
    load_results,
    paired_comparison,
    select_per_prompt,
    summarise,
)
from samed.cli import analyse as analyse_cli


def _candidates(**overrides) -> list[dict]:
    """One prompt's three candidates, mis-ranked by the model's quality head."""
    base = {
        "dataset": "chaos", "modality": "CT", "target": "liver", "subject": "CT-1",
        "patient": "CT-1",
        "image_id": "img0", "slice_index": 0, "label_value": 255, "model": "sam_vit_b",
        "strategy": "S2", "jitter": "none", "seed": 0,
        "gt_area": 1000, "pred_area": 1000,
    }
    base.update(overrides)
    rows = []
    for candidate, (dice, iou) in enumerate([(0.30, 0.90), (0.95, 0.20), (0.50, 0.50)]):
        rows.append({**base, "candidate": candidate, "predicted_iou": iou,
                     "dice": dice, "jaccard": dice / (2 - dice),
                     "hd": 10.0 * (1 - dice), "hd95": 8.0 * (1 - dice)})
    return rows


@pytest.fixture
def results(tmp_path: Path) -> Path:
    rows = _candidates()
    rows += _candidates(image_id="img1", strategy="S5")
    path = tmp_path / "shard-0000.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_load_reads_every_shard_in_a_directory(results):
    assert len(load_results(results)) == 6
    assert len(load_results(results / "shard-0000.csv")) == 6


def test_load_rejects_a_table_without_the_needed_columns(tmp_path):
    (tmp_path / "shard-0000.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_results(tmp_path)


def test_load_complains_when_there_is_nothing_to_read(tmp_path):
    with pytest.raises(FileNotFoundError, match="no result shards"):
        load_results(tmp_path)


def test_each_rule_picks_its_own_candidate(results):
    selected = select_per_prompt(load_results(results))
    assert len(selected) == 2, "one row per prompt, not per candidate"

    row = selected[selected["image_id"] == "img0"].iloc[0]
    assert row["dice_oracle"] == pytest.approx(0.95)   # best against ground truth
    assert row["dice_score"] == pytest.approx(0.30)    # highest predicted IoU
    assert row["dice_first"] == pytest.approx(0.30)    # candidate 0
    assert row["oracle_gap"] == pytest.approx(0.65)


def test_distance_metrics_follow_the_chosen_candidate(results):
    row = select_per_prompt(load_results(results)).iloc[0]
    assert row["hd_oracle"] == pytest.approx(10.0 * (1 - 0.95))
    assert row["hd_score"] == pytest.approx(10.0 * (1 - 0.30))


def test_the_oracle_is_never_beaten_by_a_deployable_rule():
    rng = np.random.default_rng(0)
    rows = []
    for prompt in range(40):
        for candidate in range(3):
            rows.append({
                **{key: "x" for key in PROMPT_KEYS},
                "image_id": f"img{prompt}", "candidate": candidate,
                "predicted_iou": rng.random(), "dice": rng.random(),
                "jaccard": 0.0, "hd": 0.0, "hd95": 0.0,
            })
    selected = select_per_prompt(pd.DataFrame(rows))
    assert (selected["oracle_gap"] >= -1e-12).all()


def test_summary_groups_and_counts(results):
    summary = summarise(select_per_prompt(load_results(results)))
    assert set(summary["strategy"]) == {"S2", "S5"}
    assert summary["n"].sum() == 2
    assert {"dice_oracle", "dice_score", "oracle_gap", "oracle_gap_lo"} <= set(summary.columns)


def test_bootstrap_interval_brackets_the_mean():
    rng = np.random.default_rng(1)
    values = rng.normal(0.8, 0.05, size=200)
    low, high = bootstrap_ci(values, seed=3)
    assert low < values.mean() < high
    assert high - low < 0.05, "200 samples should give a tight interval"


def test_bootstrap_handles_degenerate_input():
    assert bootstrap_ci([]) == (pytest.approx(float("nan"), nan_ok=True),) * 2
    assert bootstrap_ci([0.5]) == (0.5, 0.5)


def test_bootstrap_ignores_infinite_distances():
    """A missed object gives HD of inf; it must not poison an interval."""
    low, high = bootstrap_ci([0.5, 0.6, float("inf"), 0.55], seed=0)
    assert np.isfinite(low) and np.isfinite(high)


def test_cli_writes_both_tables(results, tmp_path, capsys):
    out = tmp_path / "tables"
    assert analyse_cli.main(["--results", str(results), "--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "DICE per object-modality target" in printed
    assert "Aggregated over targets" in printed

    per_strategy = pd.read_csv(out / "per_strategy.csv")
    assert set(per_strategy["strategy"]) == {"S2", "S5"}
    assert (out / "per_target.csv").exists()


def test_overlapping_shards_are_rejected(tmp_path):
    """The exact accident this guards: shard files from two different splits.

    Deduplicating quietly would be worse than failing - the duplicated prompts
    would just carry double weight in every mean.
    """
    rows = _candidates()
    for name in ("shard-0000-of-0002.csv", "shard-0000-of-0004.csv"):
        path = tmp_path / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    with pytest.raises(ValueError, match="duplicated candidate rows"):
        load_results(tmp_path)


def test_distinct_shards_load_cleanly(tmp_path):
    for index, name in enumerate(("shard-0000-of-0002.csv", "shard-0001-of-0002.csv")):
        rows = _candidates(image_id=f"img{index}")
        path = tmp_path / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    assert len(load_results(tmp_path)) == 6


def _clustered(n_subjects: int, per_subject: int, spread: float, seed: int = 0):
    """Values that vary a lot between subjects and little within them."""
    rng = np.random.default_rng(seed)
    values, labels = [], []
    for subject in range(n_subjects):
        centre = rng.normal(0.8, spread)
        for _ in range(per_subject):
            values.append(centre + rng.normal(0, 0.001))
            labels.append(f"p{subject}")
    return values, labels


def test_cluster_bootstrap_is_wider_than_the_naive_one():
    """The correction our own criticism of the paper demands.

    Near-identical slices within a subject inflate n without adding evidence, so
    resampling masks individually reports an interval far narrower than the data
    supports.
    """
    values, labels = _clustered(n_subjects=8, per_subject=40, spread=0.08)

    naive_lo, naive_hi = bootstrap_ci(values, seed=0)
    clustered_lo, clustered_hi = bootstrap_ci(values, clusters=labels, seed=0)

    assert (clustered_hi - clustered_lo) > 4 * (naive_hi - naive_lo)


def test_cluster_bootstrap_degenerates_gracefully():
    assert bootstrap_ci([0.5, 0.6], clusters=["a", "a"]) == (pytest.approx(0.55),) * 2
    lo, hi = bootstrap_ci([], clusters=[])
    assert np.isnan(lo) and np.isnan(hi)


def test_cluster_bootstrap_ignores_infinite_values():
    lo, hi = bootstrap_ci([0.5, float("inf"), 0.7], clusters=["a", "a", "b"], seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_summary_reports_how_many_subjects_back_each_row(results):
    summary = summarise(select_per_prompt(load_results(results)))
    assert (summary["n_clusters"] == 1).all(), "the fixture has a single subject"
    assert "n_clusters" in summary.columns


def test_a_multi_model_run_gets_its_own_table(tmp_path, capsys):
    """Whether the oracle gap survives model scale is the question that decides
    how far the criticism reaches, so it needs its own comparison."""
    rows = _candidates(model="sam_vit_b") + _candidates(model="sam_vit_h")
    path = tmp_path / "shard-0000-of-0001.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert analyse_cli.main(["--results", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "By model" in printed
    assert "sam_vit_b" in printed and "sam_vit_h" in printed


def test_a_single_model_run_omits_the_comparison(results, capsys):
    assert analyse_cli.main(["--results", str(results)]) == 0
    assert "By model" not in capsys.readouterr().out


def _two_models(n_prompts: int, oracle_gain: float, score_gain: float) -> pd.DataFrame:
    """Prompt-matched results where the two rules can disagree about the sign."""
    rng = np.random.default_rng(0)
    rows = []
    for prompt in range(n_prompts):
        difficulty = rng.normal(0, 0.25)   # some organs are simply harder
        for model, gain_o, gain_s in (("base", 0.0, 0.0),
                                      ("big", oracle_gain, score_gain)):
            for candidate in range(2):
                rows.append({
                    **{key: "x" for key in PROMPT_KEYS},
                    "model": model, "patient": f"p{prompt % 8}",
                    "image_id": f"img{prompt}", "candidate": candidate,
                    "predicted_iou": 1.0 - candidate,
                    "dice": 0.7 + difficulty + (gain_o if candidate else gain_s),
                    "jaccard": 0.0, "hd": 0.0, "hd95": 0.0,
                })
    return select_per_prompt(pd.DataFrame(rows))


def test_pairing_removes_the_variance_from_organ_difficulty():
    """Unpaired means are swamped by how hard each organ is; the paired
    difference is not, which is why the comparison is done prompt by prompt."""
    selected = _two_models(200, oracle_gain=0.05, score_gain=0.05)
    paired = paired_comparison(selected, baseline="base", other="big")

    assert paired["delta_dice_score"].iloc[0] == pytest.approx(0.05, abs=1e-9)
    assert paired["delta_dice_score_sig"].iloc[0] == "yes"

    spread = selected.groupby("model")["dice_score"].std().max()
    interval = paired["delta_dice_score_hi"].iloc[0] - paired["delta_dice_score_lo"].iloc[0]
    assert interval < spread, "pairing must be tighter than the raw spread"


def test_the_two_rules_can_disagree_about_which_model_is_better():
    """The finding this exists to express: a model can generate better
    candidates while ranking them worse, which reverses the verdict."""
    selected = _two_models(200, oracle_gain=0.08, score_gain=-0.06)
    paired = paired_comparison(selected, baseline="base", other="big")

    assert paired["delta_dice_oracle"].iloc[0] > 0
    assert paired["delta_dice_score"].iloc[0] < 0
    assert paired["delta_dice_score_sig"].iloc[0] == "yes"


def test_paired_comparison_needs_both_models():
    selected = _two_models(10, 0.0, 0.0)
    with pytest.raises(ValueError, match="need results for both"):
        paired_comparison(selected, baseline="base", other="absent")
