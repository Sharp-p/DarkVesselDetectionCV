"""Scientific safeguards for recall-constrained model selection and paired reports."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_recall import compare_runs, validation_runs, write_report
from tuning import combine_seed_scores, objective_score
from tuning_metrics import DetectionCurve, summarize


def records():
    return [{"gt": [[(0, 0), (20, 20), (40, 40), (60, 60)], []],
             "pred": [[(0, 0, 0.9, False), (99, 99, 0.8, True),
                       (20, 20, 0.7, False), (40, 40, 0.7, False)], []]}]


def test_target_uses_highest_feasible_cutoff_and_keeps_tied_group():
    c = DetectionCurve(records(), 0, 1)
    p = c.at_recall(0.5)
    assert p["threshold"] == 0.7
    assert p["metrics"]["recall"] == 0.75
    assert p["recall_excess"] == 0.25
    assert p["metrics"]["fp"] == p["metrics"]["near_shore_fp"] == 1
    assert c.at_recall(0.25)["threshold"] == 0.9
    assert c.at_recall(0.25, upper=0.85)["threshold"] == 0.85


def test_unreachable_floor_and_no_ground_truth_are_not_zero_fp_wins():
    c = DetectionCurve(records(), 0, 1)
    p = c.at_recall(1)
    assert p["status"] == "unreachable" and p["metrics"] is None and p["threshold"] is None
    assert p["max_recall"] == 0.75
    assert c.at_recall(0.5, lower=0.8)["status"] == "unreachable"
    assert DetectionCurve(records(), 1).at_recall(0.5)["status"] == "no_ground_truth"
    empty = [{"gt": [[(0, 0)], []], "pred": [[], []]}]
    assert DetectionCurve(empty, 0).at_recall(0.1)["status"] == "unreachable"


@pytest.mark.parametrize("target", [0, -0.1, 1.01, float("nan"), float("inf")])
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError):
        DetectionCurve(records(), 0).at_recall(target)


def test_thresholds_match_exhaustive_search_and_float_rounding():
    gt = [(i * 3, 0) for i in range(100)]
    pred = [(r, c, 1 - i/200, bool(i % 2)) for i, (r, c) in enumerate(gt)]
    c = DetectionCurve([{"gt": [gt, []], "pred": [pred, []]}], 0, 1)
    assert c.at_recall(0.14)["metrics"]["tp"] == 14
    for target in np.linspace(0.01, 1, 100):
        row = c.at_recall(float(target))["metrics"]
        brute = [r for r in c.rows if r["recall"] + 1e-12 >= target]
        assert row == max(brute, key=lambda r: r["threshold"])


def test_objective_enforces_recall_before_rewarding_low_false_alarms():
    summary = summarize(records(), distance_threshold=1, recall_targets=[0.5])
    feasible = objective_score(summary, "vessel-fp-at-recall", 0.5)
    weighted = objective_score(summary, "vessel-fp-at-recall", 0.5, coastal_fp_weight=2)
    conservative = records()
    conservative[0]["pred"][0] = conservative[0]["pred"][0][:1]
    failure = objective_score(summarize(conservative, distance_threshold=1, recall_targets=[0.5]),
                              "vessel-fp-at-recall", 0.5)
    assert 0 < weighted < feasible <= 1
    assert failure < 0  # Zero FP alone cannot beat adequate recall.
    assert combine_seed_scores([feasible, failure], "vessel-fp-at-recall") == failure


def run(predictions, seed=42):
    curves = [DetectionCurve(predictions, c, 1) for c in range(2)]
    return {"seed": seed, "contract": {"data_seed": 42, "gt": "same"},
            "points": [dict(curve.at_recall(0.5), channel=c) for c, curve in enumerate(curves)]}


def test_report_pairs_seeds_and_handles_missing_points(tmp_path):
    better = records()
    better[0]["pred"][0].pop(1)  # Remove one coastal false positive at identical recall.
    groups = {"sar": [run(records(), s) for s in [42, 43]],
              "context": [run(better, s) for s in [42, 43]]}
    report = compare_runs(groups)
    vessel = report["paired_summaries"][0]
    assert vessel["complete"] and vessel["fp_difference"]["mean"] == -1
    assert vessel["near_shore_fp_difference"]["mean"] == -1
    assert vessel["lower_mean_fp"] == "context"
    assert not report["paired_summaries"][1]["complete"]  # Structure has no GT.
    assert write_report(report, tmp_path).exists()
    assert (tmp_path/"paired_differences.csv").exists()
    json.dumps(report, allow_nan=False)


def test_report_does_not_hide_failed_seed_or_frozen_test_recall_gap():
    groups = {"sar": [run(records(), s) for s in [42, 43]],
              "context": [run(records(), s) for s in [42, 43]]}
    groups["context"][1]["points"][0].update(status="unreachable", metrics=None, threshold=None)
    summary = compare_runs(groups)["paired_summaries"][0]
    assert summary["n_comparable"] == 1 and summary["lower_mean_fp"] is None
    groups = {"sar": [run(records())], "context": [run(records())]}
    groups["context"][0]["points"][0]["metrics"]["recall"] = 0.4
    report = compare_runs(groups, split="test")
    assert report["pairs"][0]["reason"] == "recall_mismatch"
    assert report["paired_summaries"][0]["lower_mean_fp"] is None


def test_report_rejects_different_seeds_or_data():
    groups = {"sar": [run(records())], "context": [run(records())]}
    groups["context"][0]["seed"] = 43
    with pytest.raises(ValueError, match="seeds"):
        compare_runs(groups)
    groups["context"][0]["seed"] = 42
    groups["context"][0]["contract"]["data_seed"] = 43
    with pytest.raises(ValueError, match="Different data"):
        compare_runs(groups)


def test_validation_report_cannot_calibrate_test_cache(tmp_path):
    (tmp_path/"best.json").write_text(json.dumps({"runs": [{"seed": 42, "selection": "selection.json"}]}))
    (tmp_path/"selection.json").write_text(json.dumps({"threshold_source": "validation", "seed": 42, "config": {}}))
    with gzip.open(tmp_path/"validation_candidates.json.gz", "wt") as f:
        json.dump({"split": "test"}, f)
    with pytest.raises(ValueError, match="validation cache"):
        validation_runs(tmp_path/"best.json", [0.5])


