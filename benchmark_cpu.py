#!/usr/bin/env python3
"""Time candidate extraction and exact metrics on reproducible 256x256 CPU fixtures.

Optional --baseline-dir points to an extracted previous archive. This is a
microbenchmark, not a GPU training benchmark or a measurement of SAR accuracy.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

import torch
import tuning_runtime
import tuning_metrics


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(patches=4):
    generator = torch.Generator().manual_seed(123)
    logits = torch.randn(patches, 2, 256, 256, generator=generator) - 2
    inputs = torch.rand(patches, 5, 256, 256, generator=generator)
    targets = torch.zeros_like(logits)
    targets[:, :, 16:240:32, 16:240:32] = 1.0

    class Model(torch.nn.Module):
        def forward(self, inputs, return_logits=False):
            return logits.sigmoid().clamp(1e-4, 1-1e-4), logits

    return Model(), [(inputs, targets)]


def timed(fn, repeats):
    fn()  # warm imports, allocator and thread pools
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter()-start)
    return result, statistics.median(durations)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--patches", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=2)
    args = p.parse_args()
    if min(args.repeats, args.patches, args.cpu_threads) < 1:
        p.error("Repeats, patches, and CPU threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    model, batches = fixture(args.patches)
    (records, loss), extraction = timed(
        lambda: tuning_runtime.collect_candidates(model, batches, "cpu", 0.01), args.repeats)
    summary, metrics = timed(lambda: tuning_metrics.summarize(records), args.repeats)
    result = {"scope": "synthetic_CPU_microbenchmark_not_end_to_end_training",
              "patches": args.patches, "shape": [2, 256, 256], "threads": args.cpu_threads,
              "candidates": sum(len(channel) for r in records for channel in r["pred"]),
              "median_seconds": {"candidate_extraction": extraction, "exact_metrics": metrics}}
    if args.baseline_dir:
        old_runtime = load_module(args.baseline_dir/"tuning_runtime.py", "old_runtime")
        old_metrics = load_module(args.baseline_dir/"tuning_metrics.py", "old_metrics")
        (old_records, old_loss), old_extraction = timed(
            lambda: old_runtime.collect_candidates(model, batches, "cpu", 0.01), args.repeats)
        old_summary, old_metric_time = timed(lambda: old_metrics.summarize(old_records), args.repeats)
        assert records == old_records and loss == old_loss, "Candidate extraction changed"
        for key in ("best_by_channel", "fixed_0_3", "sweep", "gt_counts"):
            assert summary[key] == old_summary[key], f"Metric mismatch: {key}"
        for a, b in zip(summary["ap_at_floor"], old_summary["ap_at_floor"]):
            assert abs(a-b) < 1e-12, "AP changed beyond floating-point summation tolerance"
        result.update(equivalent=True,
                      baseline_median_seconds={"candidate_extraction": old_extraction, "exact_metrics": old_metric_time},
                      speedup={"candidate_extraction": old_extraction/extraction,
                               "exact_metrics": old_metric_time/metrics})
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
