#!/usr/bin/env python3
"""Compare false alarms at validation recall targets, or frozen test cutoffs.

No training or inference is performed. See MATCHED_RECALL_README.md.
"""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

from tuning_metrics import DetectionCurve, ground_truth_digest

METRICS = ("recall", "precision", "tp", "fp", "fn", "near_shore_fp", "f1")


def stats(values):
    return {"mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def validation_runs(manifest_path, targets):
    source = Path(manifest_path).resolve()
    manifest = json.loads(source.read_text())
    runs = []
    for run in manifest["runs"]:
        selection_path = source.parent / run["selection"]
        selection = json.loads(selection_path.read_text())
        if selection["threshold_source"] != "validation" or selection["seed"] != run["seed"]:
            raise ValueError("Invalid validation selection or seed metadata")
        cfg = selection["config"]
        with gzip.open(selection_path.parent / "validation_candidates.json.gz", "rt") as f:
            cache = json.load(f)
        if cache["split"] != "val":
            raise ValueError("Recall threshold selection requires a validation cache, never test")
        if cache["candidate_floor"] != cfg["min_score"]:
            raise ValueError("Cache candidate floor differs from selection")
        curves = [DetectionCurve(cache["records"], c, cfg["match_distance"]) for c in range(2)]
        points = [dict(curve.at_recall(target, cfg["min_score"]), channel=c)
                  for c, curve in enumerate(curves) for target in targets]
        contract = {key: cfg.get(key) for key in (
            "data_seed", "max_val_samples", "match_distance", "min_score", "smoke", "code_sha256")}
        contract.update(scenes=cfg["scenes"]["val"],
                        data_fingerprint=cfg["training_data_fingerprint"],
                        ground_truth_sha256=ground_truth_digest(cache["records"]),
                        num_patches=len(cache["records"]))
        runs.append({"seed": run["seed"], "contract": contract, "points": points,
                     "source": str(selection_path), "variant": cfg["variant"],
                     "optimizer": selection["optimizer_name"], "params": selection["params"],
                     "checkpoint_sha256": selection["checkpoint_sha256"]})
    return runs


def test_runs(directory, targets):
    runs = []
    for path in sorted(Path(directory).glob("seed_*/test_metrics.json")):
        result = json.loads(path.read_text())
        if result.get("threshold_source") != "validation":
            raise ValueError("Test operating points must use frozen validation thresholds")
        if not result.get("matched_recall"):
            raise ValueError("Test report lacks frozen recall targets; evaluate a calibrated checkpoint first")
        cfg = result["comparison_config"]
        contract = {key: cfg[key] for key in (
            "data_seed", "match_distance", "min_score", "smoke", "code_sha256")}
        contract.update(scenes=cfg["scenes"]["test"],
                        data_fingerprint=result["test_data_fingerprint"],
                        ground_truth_sha256=result["ground_truth_sha256"],
                        num_patches=result["num_patches"])
        points = [p for p in result["matched_recall"] if p["target_recall"] in targets]
        if {(p["channel"], p["target_recall"]) for p in points} != {
                (c, r) for c in range(2) for r in targets}:
            raise ValueError("Requested targets were not all frozen before test evaluation")
        runs.append({"seed": result["seed"], "contract": contract, "points": points,
                     "source": str(path), "variant": result["variant"],
                     "optimizer": result["optimizer"],
                     "checkpoint_sha256": result["checkpoint_sha256"]})
    return runs


def compare_runs(groups, split="validation", reference=None, recall_tolerance=0.005):
    """Build paired comparisons; never convert missing/unreachable points to zero FP."""
    if split not in ("validation", "test"):
        raise ValueError("Comparison split must be validation or test")
    if not math.isfinite(recall_tolerance) or not 0 <= recall_tolerance <= 1:
        raise ValueError("Recall tolerance must be in [0, 1]")
    labels = list(groups)
    if len(labels) < 2:
        raise ValueError("At least two model conditions are required")
    reference = reference or labels[0]
    if reference not in groups:
        raise ValueError("Reference label is not in the comparison")
    seeds = sorted(r["seed"] for r in groups[reference])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Require non-empty, unique seed sets")
    contracts = {r["seed"]: r["contract"] for r in groups[reference]}
    ref_points = {(p["channel"], p["target_recall"])
                  for p in groups[reference][0]["points"]}
    if not ref_points:
        raise ValueError("No recall operating points to compare")
    rows = []
    for label, runs in groups.items():
        if sorted(r["seed"] for r in runs) != seeds:
            raise ValueError("Every model must have the same unique seeds for paired comparisons")
        for run in runs:
            if run["contract"] != contracts[run["seed"]]:
                raise ValueError("Different data, patch sampling, ground truth, or metric/code contract")
            keys = [(p["channel"], p["target_recall"]) for p in run["points"]]
            if set(keys) != ref_points or len(set(keys)) != len(keys):
                raise ValueError("Every run must have the same unique channel/recall targets")
            for point in run["points"]:
                metrics = point["metrics"]
                row = {"label": label, "seed": run["seed"], "channel": point["channel"],
                       "target_recall": point["target_recall"], "status": point["status"],
                       "threshold": point["threshold"], "validation_max_recall": point["max_recall"],
                       "source": run.get("source"), "threshold_source": "validation",
                       "checkpoint_sha256": run.get("checkpoint_sha256")}
                row.update({m: metrics[m] if metrics else None for m in METRICS})
                row["target_met_on_reported_split"] = (
                    metrics["recall"] + 1e-12 >= point["target_recall"] if metrics else False)
                rows.append(row)
    by_key = {(r["label"], r["seed"], r["channel"], r["target_recall"]): r for r in rows}
    summaries, pairs, paired_summaries = [], [], []
    for channel, target in sorted(ref_points):
        for label in labels:
            subset = [by_key[label, seed, channel, target] for seed in seeds]
            valid = [r for r in subset if r["status"] == "reached" and r["recall"] is not None]
            summaries.append({"label": label, "channel": channel, "target_recall": target,
                              "n_expected": len(seeds), "n_reached": len(valid),
                              "all_validation_targets_reached": len(valid) == len(seeds),
                              **{m: stats([r[m] for r in valid]) for m in METRICS}})
            if label == reference:
                continue
            group_pairs = []
            for seed in seeds:
                a, b = by_key[label, seed, channel, target], by_key[reference, seed, channel, target]
                available = a["recall"] is not None and b["recall"] is not None
                gap = a["recall"] - b["recall"] if available else None
                comparable = available and abs(gap) <= recall_tolerance + 1e-12
                pair = {"label": label, "reference": reference, "seed": seed,
                        "channel": channel, "target_recall": target,
                        "comparable_recall": comparable, "recall_difference": gap,
                        "reason": ("within_tolerance" if comparable else
                                   "recall_mismatch" if available else "unreachable_or_no_ground_truth")}
                for m in ("fp", "near_shore_fp"):
                    pair[m + "_difference"] = a[m] - b[m] if available else None
                group_pairs.append(pair)
            pairs.extend(group_pairs)
            valid_pairs = [p for p in group_pairs if p["comparable_recall"]]
            complete = len(valid_pairs) == len(seeds)
            summary = {"label": label, "reference": reference, "channel": channel,
                       "target_recall": target, "n_expected": len(seeds),
                       "n_comparable": len(valid_pairs), "complete": complete,
                       "interpretation": "descriptive_paired_counts" if complete else "inconclusive_missing_or_mismatched_recall"}
            for m in ("fp", "near_shore_fp"):
                summary[m + "_difference"] = stats([p[m + "_difference"] for p in valid_pairs])
                difference = summary[m + "_difference"]["mean"]
                summary["lower_mean_" + m] = (None if not complete else "tie" if difference == 0
                                               else label if difference < 0 else reference)
            paired_summaries.append(summary)
    return {"schema": 1, "split": split, "threshold_source": "validation",
            "scope": "sampled_patch_instances_not_scene_level", "reference": reference,
            "recall_tolerance": recall_tolerance, "seeds": seeds, "labels": labels,
            "coastal_definition": "normalized_distance_to_shore <= 0.1 (nominally 5 km)",
            "note": ("Validation-selected points are descriptive; score ties may overshoot targets."
                     if split == "validation" else
                     "Cutoffs frozen on validation; actual test recall can differ. No test threshold tuning."),
            "provenance": {label: [{k: v for k, v in run.items() if k != "points"} for run in runs]
                           for label, runs in groups.items()},
            "rows": rows, "summaries": summaries, "pairs": pairs, "paired_summaries": paired_summaries}


def write_report(report, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "matched_recall.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for key, filename in (("rows", "matched_recall.csv"), ("pairs", "paired_differences.csv")):
        with (out / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report[key][0]))
            writer.writeheader()
            writer.writerows(report[key])
    def display(stat):
        if stat["mean"] is None:
            return "unreachable / no GT"
        return f"{stat['mean']:.3f}" + (f" ± {stat['std']:.3f}" if stat["std"] is not None else " (one seed)")
    lines = ["# False positives at validation recall targets", "", report["note"], "",
             f"Split: **{report['split']}**. Reference: **{report['reference']}**. "
             f"Allowed recall difference: {report['recall_tolerance']:.4f}.", "",
             "Counts are sampled patch instances, not deduplicated scene detections. "
             "Coastal FP uses normalized shore distance ≤ 0.1 (nominally 5 km).", "",
             "| Condition | Class | Target R | Seeds with cutoff | Actual R | FP | Coastal FP |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for s in report["summaries"]:
        lines.append(f"| {s['label']} | {('vessel', 'structure')[s['channel']]} | {s['target_recall']:.2f} | "
                     f"{s['n_reached']}/{s['n_expected']} | {display(s['recall'])} | "
                     f"{display(s['fp'])} | {display(s['near_shore_fp'])} |")
    lines += ["", "Incomplete rows show available seeds only; do not rank them against complete rows.", "",
              "## Paired differences against the reference", "",
              "Negative differences mean fewer false positives. Only pairs within the stated "
              "recall tolerance enter these averages. Sample SD describes seed variation, not a confidence interval.", "",
              "| Condition | Class | Target R | Comparable seeds | Δ FP | Δ Coastal FP | Status |",
              "|---|---|---:|---:|---:|---:|---|"]
    for s in report["paired_summaries"]:
        lines.append(f"| {s['label']} | {('vessel', 'structure')[s['channel']]} | {s['target_recall']:.2f} | "
                     f"{s['n_comparable']}/{s['n_expected']} | {display(s['fp_difference'])} | "
                     f"{display(s['near_shore_fp_difference'])} | "
                     f"{'descriptive comparison' if s['complete'] else 'inconclusive'} |")
    (out / "MATCHED_RECALL_REPORT.md").write_text("\n".join(lines) + "\n")
    return out / "MATCHED_RECALL_REPORT.md"


def compare_manifests(paths, labels, output, targets=(0.5, 0.6, 0.7), reference=None, recall_tolerance=0.005):
    if len(paths) != len(labels) or len(set(labels)) != len(labels):
        raise ValueError("Supply one unique label per manifest")
    groups = {label: validation_runs(path, sorted(set(targets))) for label, path in zip(labels, paths)}
    return write_report(compare_runs(groups, "validation", reference, recall_tolerance), output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--best", nargs="+", help="Selected best.json manifests; uses validation caches only")
    source.add_argument("--test-reports", nargs="+", help="tuning.py evaluate output directories")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--reference", help="Reference label; default is the first label")
    p.add_argument("--targets", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    p.add_argument("--recall-tolerance", type=float, default=0.005,
                   help="Maximum absolute achieved-recall gap for a paired FP comparison")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if any(not math.isfinite(t) or not 0 < t <= 1 for t in args.targets):
        p.error("Targets must be in (0, 1]")
    paths = args.best or args.test_reports
    if len(paths) != len(args.labels) or len(set(args.labels)) != len(args.labels):
        p.error("Supply one unique label per input")
    if args.best:
        report_path = compare_manifests(paths, args.labels, args.output, args.targets,
                                       args.reference, args.recall_tolerance)
    else:
        groups = {label: test_runs(path, args.targets) for label, path in zip(args.labels, paths)}
        report_path = write_report(compare_runs(groups, "test", args.reference, args.recall_tolerance), args.output)
    print(f"Matched-recall report: {report_path}")


if __name__ == "__main__":
    main()
