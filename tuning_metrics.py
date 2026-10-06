"""Detection metrics on cached patch candidates. No torch/Optuna dependency.

Greedy confidence-priority matching is identical in meaning to utils.py.
AP = sum over distinct scores of (recall increase * precision), without an
interpolated upper envelope. It is truncated at the candidate confidence floor.
Metrics count patch instances; overlapping patches are NOT deduplicated scenes.
"""
from bisect import bisect_right
from itertools import groupby
import math


def _metrics(tp, fp, total_gt, threshold, near_shore_fp=0):
    fn = total_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return dict(threshold=float(threshold), tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1,
                near_shore_fp=near_shore_fp,
                near_shore_fp_fraction_of_fp=near_shore_fp / fp if fp else 0.0)


class DetectionCurve:
    def __init__(self, records, channel, distance_threshold=20.0):
        events, total_gt = [], 0
        for record in records:
            gt = record["gt"][channel]
            total_gt += len(gt)
            unmatched = set(range(len(gt)))
            # Stable row/column order breaks score ties within each patch.
            preds = sorted(record["pred"][channel], key=lambda p: (-p[2], p[0], p[1]))
            for row, col, score, near_shore in preds:
                nearest = min(unmatched, key=lambda i: (math.hypot(row-gt[i][0], col-gt[i][1]), i), default=None)
                is_tp = nearest is not None and math.hypot(row-gt[nearest][0], col-gt[nearest][1]) <= distance_threshold
                if is_tp:
                    unmatched.remove(nearest)
                events.append((float(score), int(is_tp), int(not is_tp and near_shore)))
        self.total_gt = total_gt
        self.rows, self.negative_scores = [], []
        tp = fp = coastal = 0
        ap = 0.0
        for score, tied in groupby(sorted(events, key=lambda e: -e[0]), key=lambda e: e[0]):
            group_tp = group_fp = 0
            for _, hit, near in tied:
                group_tp += hit
                group_fp += 1 - hit
                coastal += near
            tp += group_tp
            fp += group_fp
            row = _metrics(tp, fp, total_gt, score, coastal)
            self.rows.append(row)
            self.negative_scores.append(-score)
            if total_gt:
                ap += (group_tp / total_gt) * row["precision"]
        self.ap = ap if total_gt else None

    def at(self, threshold):
        idx = bisect_right(self.negative_scores, -threshold) - 1
        if idx < 0:
            return _metrics(0, 0, self.total_gt, threshold)
        result = dict(self.rows[idx])
        result["threshold"] = float(threshold)
        return result

    def best_f1(self, lower, upper):
        candidates = [self.at(lower), self.at(upper)]
        candidates.extend(r for r in self.rows if lower <= r["threshold"] <= upper)
        # On equal F1 prefer fewer false positives, then the higher cutoff.
        return dict(max(candidates, key=lambda r: (r["f1"], -r["fp"], r["threshold"])))


def summarize(records, min_score=0.01, threshold_min=0.01, threshold_max=0.95,
              thresholds=None, distance_threshold=20.0):
    if not records:
        raise ValueError("No evaluation patches")
    if not 0 < min_score <= threshold_min <= threshold_max < 1:
        raise ValueError("Require 0 < min_score <= threshold_min <= threshold_max < 1")
    if min_score > 0.3:
        raise ValueError("min_score must be <= 0.3 for the fixed-threshold comparison")
    curves = [DetectionCurve(records, c, distance_threshold) for c in range(2)]
    thresholds = thresholds or [min_score] + [i/100 for i in range(1, 100) if i/100 >= min_score]
    if any(t < min_score or not 0 < t < 1 for t in thresholds):
        raise ValueError("Sweep thresholds must be >= the cached candidate floor and < 1")
    return {
        "scope": "sampled_patch_instances_not_scene_level",
        "num_patches": len(records), "candidate_floor": min_score,
        "ap_definition": "non_interpolated_grouped_score_detection_AP_above_candidate_floor",
        "ap_at_floor": [c.ap for c in curves],
        "gt_counts": [c.total_gt for c in curves],
        "best_by_channel": [c.best_f1(threshold_min, threshold_max) for c in curves],
        "fixed_0_3": [c.at(0.3) for c in curves],
        "sweep": [{"threshold": t, "metrics": [c.at(t) for c in curves]} for t in sorted(set(thresholds))],
    }


def at_frozen_thresholds(records, thresholds, distance_threshold=20.0):
    if len(thresholds) != 2:
        raise ValueError("Exactly two class thresholds are required")
    return [DetectionCurve(records, c, distance_threshold).at(float(thresholds[c])) for c in range(2)]
