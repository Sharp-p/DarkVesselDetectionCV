"""Detection metrics on cached patch candidates. No torch/Optuna dependency.

Greedy confidence-priority matching is identical in meaning to utils.py.
AP = sum over distinct scores of (recall increase * precision), without an
interpolated upper envelope. It is truncated at the candidate confidence floor.
Metrics count patch instances; overlapping patches are NOT deduplicated scenes.
"""
import math
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree


def ground_truth_digest(records):
    """Order-sensitive signature; combine with data/sampling provenance for comparisons."""
    return hashlib.sha256(json.dumps([r["gt"] for r in records],
                                    separators=(",", ":")).encode()).hexdigest()


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
    """Exact greedy matching with spatial indexing and compact cumulative arrays.

    No candidate truncation, approximate neighbors, score rounding, or binning.
    Python matching only inspects GT within the allowed distance, while grouped
    precision/recall accumulation runs in NumPy instead of allocating one dict
    for every distinct score.
    """
    def __init__(self, records, channel, distance_threshold=20.0):
        score_parts, hit_parts, coastal_parts = [], [], []
        total_gt = 0
        for record in records:
            gt = record["gt"][channel]
            total_gt += len(gt)
            raw = record["pred"][channel]
            if not len(raw):
                continue
            preds = np.asarray(raw, dtype=np.float64)
            preds = preds[np.lexsort((preds[:, 1], preds[:, 0], -preds[:, 2]))]
            hits = np.zeros(len(preds), dtype=np.int64)
            if len(gt) and distance_threshold >= 0:
                coordinates = np.asarray(gt, dtype=np.float64)
                tree = cKDTree(coordinates)
                # Slightly inclusive lookup followed by the original hypot check
                # preserves matching at an exact distance boundary.
                neighbors = tree.query_ball_point(preds[:, :2],
                    np.nextafter(float(distance_threshold), np.inf), eps=0, workers=1,
                    return_sorted=True)
                used, matched = np.zeros(len(gt), dtype=bool), 0
                for j, candidates in enumerate(neighbors):
                    if matched == len(gt):
                        break
                    if not candidates:
                        continue
                    row, col = preds[j, :2]
                    nearest = min((i for i in candidates if not used[i]),
                        key=lambda i: (math.hypot(row-gt[i][0], col-gt[i][1]), i), default=None)
                    if nearest is not None and math.hypot(row-gt[nearest][0], col-gt[nearest][1]) <= distance_threshold:
                        used[nearest] = True
                        hits[j] = 1
                        matched += 1
            score_parts.append(preds[:, 2])
            hit_parts.append(hits)
            coastal_parts.append((1-hits)*preds[:, 3].astype(np.int64))
        self.total_gt = total_gt
        if score_parts:
            scores = np.concatenate(score_parts)
            order = np.argsort(-scores, kind="stable")
            scores = scores[order]
            hits = np.concatenate(hit_parts)[order]
            coastal = np.concatenate(coastal_parts)[order]
            ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores)-1]
            self.scores = scores[ends]
            self.tp = np.cumsum(hits)[ends]
            self.fp = ends + 1 - self.tp
            self.coastal = np.cumsum(coastal)[ends]
        else:
            self.scores = np.empty(0, dtype=np.float64)
            self.tp = self.fp = self.coastal = np.empty(0, dtype=np.int64)
        self.negative_scores = -self.scores
        self.precision = self.tp / np.maximum(self.tp + self.fp, 1)
        self.f1 = 2*self.tp / np.maximum(self.tp + self.fp + total_gt, 1)
        self.ap = (float(np.sum(np.diff(np.r_[0, self.tp])/total_gt*self.precision))
                   if total_gt else None)

    def _row(self, index, threshold=None):
        return _metrics(int(self.tp[index]), int(self.fp[index]), self.total_gt,
                        self.scores[index] if threshold is None else threshold,
                        int(self.coastal[index]))

    @property
    def rows(self):
        # Compatibility for external callers; normal metric paths stay compact.
        return [self._row(i) for i in range(len(self.scores))]

    def at(self, threshold):
        idx = int(np.searchsorted(self.negative_scores, -threshold, side="right")) - 1
        return self._row(idx, threshold) if idx >= 0 else _metrics(0, 0, self.total_gt, threshold)

    def best_f1(self, lower, upper):
        candidates = [self.at(lower), self.at(upper)]
        eligible = np.flatnonzero((self.scores >= lower) & (self.scores <= upper))
        if len(eligible):
            best = eligible[self.f1[eligible] == self.f1[eligible].max()]
            best = best[self.fp[best] == self.fp[best].min()]
            # Scores are descending; first remaining index is the highest cutoff.
            candidates.append(self._row(int(best[0])))
        return dict(max(candidates, key=lambda r: (r["f1"], -r["fp"], r["threshold"])))

    def at_recall(self, target, lower=0.01, upper=1.0):
        """Highest deployable cutoff reaching target; no interpolation or tie splitting.

        At a fixed checkpoint, cumulative total/coastal FP cannot decrease as
        the cutoff is lowered. The first feasible score therefore minimizes
        both counts among the available threshold operating points.
        """
        if not math.isfinite(target) or not 0 < target <= 1:
            raise ValueError("Recall targets must be finite and in (0, 1]")
        if not 0 < lower <= upper <= 1:
            raise ValueError("Invalid recall threshold bounds")
        maximum = self.at(lower)
        result = {"target_recall": float(target), "status": "unreachable",
                  "threshold": None, "metrics": None,
                  "max_recall": maximum["recall"], "recall_excess": None}
        if not self.total_gt:
            return dict(result, status="no_ground_truth")
        # nextafter avoids e.g. 0.14 * 100 rounding just above 14.
        needed = math.ceil(np.nextafter(target * self.total_gt, -np.inf))
        if maximum["tp"] < needed:
            return result
        row = self.at(upper)
        if row["tp"] < needed:
            index = int(np.searchsorted(self.tp, needed, side="left"))
            row = self._row(index)
        return dict(result, status="reached", threshold=row["threshold"], metrics=row,
                    recall_excess=max(0.0, row["recall"]-target))


def summarize(records, min_score=0.01, threshold_min=0.01, threshold_max=0.95,
              thresholds=None, distance_threshold=20.0, recall_targets=(0.5, 0.6, 0.7)):
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
        "matched_recall": [dict(curve.at_recall(target, min_score), channel=channel)
                           for channel, curve in enumerate(curves)
                           for target in sorted(set(recall_targets))],
        "sweep": [{"threshold": t, "metrics": [c.at(t) for c in curves]} for t in sorted(set(thresholds))],
    }


def at_frozen_thresholds(records, thresholds, distance_threshold=20.0):
    if len(thresholds) != 2:
        raise ValueError("Exactly two class thresholds are required")
    return [DetectionCurve(records, c, distance_threshold).at(float(thresholds[c])) for c in range(2)]
