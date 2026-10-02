"""Detection metrics on cached patch candidates. No torch/Optuna dependency.

Greedy confidence-priority matching is identical in meaning to utils.py.
AP = sum over distinct scores of (recall increase * precision), without an
interpolated upper envelope. It is truncated at the candidate confidence floor.
Metrics count patch instances; overlapping patches are NOT deduplicated scenes.
"""
import numpy as np


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
        scores, hits, near_fp = [], [], []
        total_gt = 0
        for record in records:
            gt = np.asarray(record["gt"][channel], dtype=np.float64).reshape(-1, 2)
            preds = np.asarray(record["pred"][channel], dtype=np.float64).reshape(-1, 4)
            total_gt += len(gt)
            if not len(preds):
                continue
            # Stable row/column order breaks score ties within each patch.
            preds = preds[np.lexsort((preds[:, 1], preds[:, 0], -preds[:, 2]))]
            hit = np.zeros(len(preds), dtype=bool)
            unmatched = np.ones(len(gt), dtype=bool)
            # Bound temporary distance matrices. Only nearby predictions enter
            # the sequential greedy loop; background false positives stay arrays.
            chunk_size = max(1, min(2048, 262144 // max(1, len(gt))))
            for start in range(0, len(preds), chunk_size):
                if not unmatched.any():
                    break
                chunk = preds[start:start + chunk_size]
                distances = np.hypot(chunk[:, 0, None] - gt[None, :, 0],
                                     chunk[:, 1, None] - gt[None, :, 1])
                distances[:, ~unmatched] = np.inf
                for index in np.flatnonzero((distances <= distance_threshold).any(axis=1)):
                    row = distances[index]
                    row[~unmatched] = np.inf
                    nearest = int(row.argmin())  # first GT index wins a distance tie
                    if unmatched[nearest] and row[nearest] <= distance_threshold:
                        hit[start + index] = True
                        unmatched[nearest] = False
                    if not unmatched.any():
                        break
            scores.append(preds[:, 2])
            hits.append(hit)
            near_fp.append(~hit & preds[:, 3].astype(bool))
        self.total_gt = total_gt
        values = np.concatenate(scores) if scores else np.empty(0)
        order = np.argsort(-values, kind="stable")
        values = values[order]
        # Evaluate at the end of each tied score group, exactly as before.
        ends = (np.r_[np.flatnonzero(values[:-1] != values[1:]), len(values) - 1]
                if len(values) else np.empty(0, dtype=np.int64))
        self.negative_scores = -values[ends]
        hit = np.concatenate(hits)[order] if hits else np.empty(0, dtype=bool)
        coastal = np.concatenate(near_fp)[order] if near_fp else np.empty(0, dtype=bool)
        self._tp = np.cumsum(hit, dtype=np.int64)[ends]
        self._fp = ends + 1 - self._tp
        self._coastal = np.cumsum(coastal, dtype=np.int64)[ends]
        increments = np.diff(np.r_[0, self._tp])
        self.ap = (float(np.sum((increments / total_gt) * self._tp / (ends + 1)))
                   if total_gt else None)

    def _row(self, index, threshold):
        return _metrics(int(self._tp[index]), int(self._fp[index]), self.total_gt,
                        threshold, int(self._coastal[index]))

    @property
    def rows(self):
        """Legacy inspection API; summaries avoid materializing per-score dicts."""
        return [self._row(i, -s) for i, s in enumerate(self.negative_scores)]

    def at(self, threshold):
        idx = int(np.searchsorted(self.negative_scores, -threshold, side="right")) - 1
        if idx < 0:
            return _metrics(0, 0, self.total_gt, threshold)
        return self._row(idx, threshold)

    def best_f1(self, lower, upper):
        candidates = [self.at(lower), self.at(upper)]
        indices = np.flatnonzero((-self.negative_scores >= lower) & (-self.negative_scores <= upper))
        if len(indices):
            tp, fp = self._tp[indices], self._fp[indices]
            f1 = 2.0 * tp / (tp + fp + self.total_gt)
            # Scores descend, FP counts never decrease: the first maximum also
            # wins both existing tie-breaks (fewer FP, then higher threshold).
            index = int(indices[np.argmax(f1)])
            candidates.append(self._row(index, -self.negative_scores[index]))
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
