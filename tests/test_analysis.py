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
    attribute_correlations,
    load_attributes,
    load_results,
    merge_attributes,
    paired_comparison,
    partial_spearman,
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
    path = tmp_path / "prompted-shard-0000-of-0001.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_load_reads_every_shard_in_a_directory(results):
    assert len(load_results(results)) == 6
    assert len(load_results(results / "prompted-shard-0000-of-0001.csv")) == 6


def test_load_rejects_a_table_without_the_needed_columns(tmp_path):
    (tmp_path / "prompted-shard-0000-of-0001.csv").write_text("a,b\n1,2\n")
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
    for name in ("prompted-shard-0000-of-0002.csv", "prompted-shard-0000-of-0004.csv"):
        path = tmp_path / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    with pytest.raises(ValueError, match="duplicated candidate rows"):
        load_results(tmp_path)


def test_distinct_shards_load_cleanly(tmp_path):
    for index, name in enumerate(("prompted-shard-0000-of-0002.csv", "prompted-shard-0001-of-0002.csv")):
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
    path = tmp_path / "prompted-shard-0000-of-0001.csv"
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


def test_the_s1_caveat_is_printed_only_when_s1_is_present(tmp_path, capsys):
    """S1's gap has a different meaning and must not be read as a ranking failure."""
    for strategy, expected in (("S1", True), ("S5", False)):
        rows = _candidates(strategy=strategy)
        path = tmp_path / f"everything-shard-0000-of-0001-{strategy}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        assert analyse_cli.main(["--results", str(path)]) == 0
        assert ("S1 is not comparable" in capsys.readouterr().out) is expected


# --------------------------------------------------------------------------- #
# Object attributes and Table 6
# --------------------------------------------------------------------------- #


def _attribute_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """`contrast` drives the outcome; `size` only correlates with contrast.

    A raw correlation credits size with contrast's effect. A partial one must
    not - which is the whole reason the paper uses partial correlations, since
    object size and boundary complexity are themselves related.
    """
    rng = np.random.default_rng(seed)
    contrast = rng.normal(0, 1, n)
    size = contrast * 0.9 + rng.normal(0, 0.4, n)
    return pd.DataFrame({
        "dice_score": 0.5 + 0.3 * contrast + rng.normal(0, 0.05, n),
        "intensity_difference": contrast,
        "area": size,
        "fourier_paper": rng.normal(0, 1, n),
        "modality_code": rng.integers(0, 3, n),
        "aspect_ratio": rng.uniform(0.2, 1.0, n),
        "patient": [f"p{i % 20}" for i in range(n)],
        "strategy": "S5",
    })


def test_partial_correlation_does_not_credit_a_confounded_predictor():
    frame = _attribute_frame()
    partial = partial_spearman(frame, "dice_score", ["intensity_difference", "area"])

    raw_size = frame["dice_score"].corr(frame["area"], method="spearman")
    assert raw_size > 0.5, "size looks influential before controlling for contrast"
    assert partial["intensity_difference"] > 0.5
    assert abs(partial["area"]) < 0.2, "after controlling, size adds almost nothing"


def test_partial_correlation_survives_a_constant_predictor():
    frame = _attribute_frame(50)
    frame["aspect_ratio"] = 1.0
    result = partial_spearman(frame, "dice_score", ["intensity_difference", "aspect_ratio"])
    assert np.isnan(result["aspect_ratio"])
    assert np.isfinite(result["intensity_difference"])


def test_attribute_correlations_flag_what_the_data_supports():
    frame = _attribute_frame()
    table = attribute_correlations(frame, resamples=100)

    assert list(table["strategy"]) == ["S5"]
    assert table["intensity_difference"].iloc[0] > 0.5
    assert table["intensity_difference_sig"].iloc[0] == "yes"
    assert table["aspect_ratio_sig"].iloc[0] == "no", "a pure noise predictor must not pass"


def test_merge_attributes_needs_a_shared_manifest(results, tmp_path):
    selected = select_per_prompt(load_results(results))
    attributes = pd.DataFrame([{
        "dataset": "elsewhere", "modality": "CT", "target": "liver",
        "image_id": "other", "label_value": 255, "area": 10,
    }])
    with pytest.raises(ValueError, match="share no objects"):
        merge_attributes(selected, attributes)


def test_load_attributes_codes_the_modality(tmp_path):
    rows = [
        {"dataset": "chaos", "modality": m, "target": "liver", "image_id": f"img{i}",
         "label_value": 255, "area": 100 + i, "aspect_ratio": 0.5,
         "intensity_difference": 20.0, "fourier_paper": 5.0, "fourier_corrected": 11.0}
        for i, m in enumerate(["CT", "T1W-MRI", "T2W-MRI"])
    ]
    path = tmp_path / "attributes-shard-0000-of-0001.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    frame = load_attributes(tmp_path)
    assert sorted(frame["modality_code"].unique()) == [0, 1, 2]
    assert len(frame) == 3


def _jitter_results(tmp_path: Path) -> Path:
    """Clean and perturbed runs, as the jitter study produces them."""
    for level, penalty in (("none", 0.0), ("10-20", 0.2)):
        rows = []
        for prompt in range(6):
            for candidate, (d, iou) in enumerate([(0.9 - penalty, 0.8), (0.4, 0.3)]):
                rows.append({
                    **{key: "x" for key in PROMPT_KEYS},
                    "strategy": "S5", "jitter": level, "patient": f"p{prompt % 3}",
                    "image_id": f"img{prompt}", "candidate": candidate,
                    "predicted_iou": iou, "dice": d,
                    "jaccard": 0.0, "hd": 0.0, "hd95": 0.0,
                })
        directory = tmp_path / f"jitter-{level}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "prompted-shard-0000-of-0001.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return tmp_path


def test_perturbed_runs_do_not_pollute_the_headline_tables(tmp_path, capsys):
    """Jitter results land beside the clean ones; averaging them would quietly
    depress every published number."""
    root = _jitter_results(tmp_path)
    assert analyse_cli.main(["--results", str(root)]) == 0

    printed = capsys.readouterr().out
    assert "tables below describe 'none' only" in printed
    assert "Prompt perturbation" in printed

    from samed.analysis import load_results as load
    everything = select_per_prompt(load(root))
    clean_only = everything[everything["jitter"] == "none"]
    assert clean_only["dice_score"].mean() > everything["dice_score"].mean()


def test_an_unknown_jitter_level_is_refused(tmp_path, capsys):
    root = _jitter_results(tmp_path)
    assert analyse_cli.main(["--results", str(root), "--jitter", "40-50"]) == 2
    assert "no results at jitter level" in capsys.readouterr().out


def test_jitter_can_be_pooled_on_request(tmp_path):
    root = _jitter_results(tmp_path)
    assert analyse_cli.main(["--results", str(root), "--jitter", "all"]) == 0


def test_the_oracle_rule_flattens_attribute_dependence():
    """The mechanism behind the gap: picking the best candidate by ground truth
    is worth most where the object is hard, so it damps exactly the relationship
    the paper's Table 6 sets out to measure."""
    frame = _attribute_frame(600)
    # A deployable rule tracks the object's contrast; an oracle rule recovers a
    # good mask regardless, so its scores depend on contrast far less.
    frame["oracle_gap"] = np.clip(0.4 - 0.3 * frame["intensity_difference"], 0, None)
    frame["dice_oracle"] = frame["dice_score"] + frame["oracle_gap"]

    deployable = attribute_correlations(frame, outcome="dice_score", resamples=50)
    oracle = attribute_correlations(frame, outcome="dice_oracle", resamples=50)
    gap = attribute_correlations(frame, outcome="oracle_gap", resamples=50)

    assert oracle["intensity_difference"].iloc[0] < deployable["intensity_difference"].iloc[0]
    assert gap["intensity_difference"].iloc[0] < 0, "the gap grows as contrast falls"


def test_the_perturbation_reading_is_carried_in_the_data(tmp_path):
    """Encoding it only in the filename makes the two readings indistinguishable
    once the shards are concatenated - and the duplicate check then rejects
    everything, because the rows are genuinely identical on every key."""
    from samed.analysis import LEGACY_DEFAULTS

    rows = _candidates(jitter="10-20")
    path = tmp_path / "prompted-shard-0000-of-0001-readingfirstshift.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="encodes a perturbation reading in its"):
        load_results(tmp_path)

    assert set(LEGACY_DEFAULTS) <= set(PROMPT_KEYS)


def test_older_files_without_the_reading_take_the_defaults(results):
    frame = load_results(results)
    assert set(frame["jitter_points"]) == {"all"}
    assert set(frame["jitter_box_mode"]) == {"perturb"}


def test_two_readings_of_the_same_prompt_are_not_duplicates(tmp_path):
    for points, box_mode in (("all", "perturb"), ("first", "shift")):
        rows = _candidates(jitter="20-30", jitter_points=points, jitter_box_mode=box_mode)
        path = tmp_path / f"prompted-shard-0000-of-0001-reading{points}{box_mode}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    frame = load_results(tmp_path)
    assert len(frame) == 6
    assert set(frame["jitter_points"]) == {"all", "first"}


def test_defaults_reach_older_files_even_beside_newer_ones(tmp_path):
    """Filling on the concatenated frame silently leaves the old rows NaN as
    soon as one newer file supplies the column."""
    old = _candidates(jitter="10-20")           # no reading columns at all
    path = tmp_path / "prompted-shard-0000-of-0002.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(old[0]))
        writer.writeheader()
        writer.writerows(old)

    new = _candidates(jitter="10-20", image_id="img9",
                      jitter_points="first", jitter_box_mode="shift")
    path = tmp_path / "prompted-shard-0001-of-0002-readingfirstshift.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(new[0]))
        writer.writeheader()
        writer.writerows(new)

    frame = load_results(tmp_path)
    assert not frame["jitter_points"].isna().any()
    assert set(frame["jitter_points"]) == {"all", "first"}
