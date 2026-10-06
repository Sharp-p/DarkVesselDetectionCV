# Compare false positives at the same recall

This update addresses the concern that fewer false alarms may simply mean fewer
detections. It adds exact recall threshold selection, a recall-constrained
checkpoint/Optuna objective, and automatic paired reports across seeds. The model
architecture and existing CPU optimizations are unchanged. No finer-resolution
experiment is included. Detection improvements still require real-data testing.

## Install

Use the Linux environment and dependencies in **TUNING_README.md**; no additional
packages are needed. Activate your environment from the extracted project folder:

```bash
source .venv/bin/activate
python -m pytest -q tests
```

Use a fresh output directory and study prefix after this update. Existing studies
have a code fingerprint and cannot safely resume under changed selection rules.

## One command: wind versus no wind

```bash
python compare_cgcam.py --wind-only --groups local \
  --data-dir /absolute/path/to/dataset --output recall_wind_comparison \
  --study-prefix recall-wind-v1 \
  --optimizer rmsprop --lr 0.0002649149454284989 --weight-decay 0 \
  --objective vessel-fp-at-recall --target-recall 0.6 \
  --recall-targets 0.5 0.6 0.7 --coastal-fp-weight 0 \
  --seeds 777 888 999 --epochs 30 --device cuda --workers 4
```

This trains **2 variants × 3 seeds × 30 epochs = 180 total epochs**. It uses the
existing local gate, identical optimizer settings, seeds, initialization, splits,
sampling, scheduler and training budget for both variants. To compare with SAR-only
and early fusion too, omit `--wind-only` and use `--groups control local`. That
is six configurations, 18 runs and 540 total epochs with these three seeds.

RMSprop here is a fixed control, not a claim that it is best. All five optimizers
remain available. For an installation check, add `--smoke --epochs 1 --seeds 42
--device cpu --workers 0`, and choose an output and prefix starting with `smoke`.
Synthetic scores have no scientific meaning.

## How model selection changes

`vessel-fp-at-recall` chooses checkpoints and Optuna trials using the vessel class:

1. Find the highest score cutoff achieving the requested **validation recall**.
2. If that target cannot be reached above the saved candidate floor, mark it
   unreachable. Missing points have null metrics, never zero false positives.
3. Among feasible models, minimize `FP + coastal_fp_weight × coastal_FP`.

The feasible score maximized by Optuna is `1 / (1 + cost / vessel_GT)`. An
infeasible score is `maximum_recall - target_recall`, which is negative. A trial
with any infeasible seed receives its worst seed score; otherwise seed scores
are averaged. This prevents an infeasible seed being hidden by good seeds.

The default coastal weight is zero: all false positives count equally. A weight
of one makes a coastal false positive cost twice as much as an offshore one.
Choose this weight and recall target on validation before looking at test results.
The focal training loss is unchanged; this is checkpoint/hyperparameter selection,
not a differentiable recall loss. Best-F1 metrics remain available for context.
Ordinary `tuning.py search` retains its previous F1 objective unless you explicitly
choose `--objective vessel-fp-at-recall`.

Thresholds use every distinct candidate score, not a coarse sweep. Equal scores
stay grouped, so achieved recall can exceed the target. Targets use thresholds
from `--min-score` to 1; the F1-only `--threshold-min/max` bounds do not restrict
this search. Unreachable means unreachable within the saved candidate pool.

## Automatic reports

After training, `compare_cgcam.py` creates a `*_matched_recall_*` directory with:

- `MATCHED_RECALL_REPORT.md`: readable per-class tables and paired differences.
- `matched_recall.csv`: cutoff, actual recall, TP, FP, FN, coastal FP and F1 per seed.
- `paired_differences.csv`: FP/coastal-FP differences and actual recall gaps.
- `matched_recall.json`: full report, statuses and provenance.

The reference is `control/sar_only` when present, otherwise the first condition.
Only pairs with actual recall within `--recall-tolerance` enter paired averages
(default 0.005, half a percentage point). Use zero for strictly equal recall.
If a seed is missing, unreachable, has no class GT or has a larger recall gap,
the comparison is marked incomplete/inconclusive. No overall lower-FP label is
assigned unless all seeds are comparable. Means and sample SD describe seed
variation; they are not confidence intervals or evidence of significance.

To compare already selected models without training or inference:

```bash
python compare_recall.py \
  --best /path/to/sar/best.json /path/to/no_wind/best.json /path/to/wind/best.json \
  --labels sar_only no_wind wind --reference sar_only \
  --targets 0.5 0.6 0.7 --output validation_recall_report
```

This reads **validation candidate caches only**. It rejects different seeds,
datasets, sampling, GT order, score floors or metric/code contracts. It can report
new targets from validation caches but does not rewrite saved checkpoint selections.

## Final test evaluation

New training selections freeze each recall cutoff alongside the best-F1 cutoffs.
Evaluate each chosen `best.json` with the existing command, then compare reports:

```bash
python tuning.py evaluate --best /path/to/no_wind/best.json \
  --output test_no_wind --device cuda --confirm-frozen
python tuning.py evaluate --best /path/to/wind/best.json \
  --output test_wind --device cuda --confirm-frozen
python compare_recall.py --test-reports test_no_wind test_wind \
  --labels no_wind wind --targets 0.5 0.6 0.7 --output test_recall_report
```

These commands never tune thresholds on test. **Matching a validation recall target
does not guarantee equal test recall.** The test report shows the achieved recall,
flags mismatches and withholds a matched-recall conclusion when appropriate. The
test report command accepts only targets frozen before evaluation; choose the
same `--recall-targets` before training/calibration and during reporting.

For an existing checkpoint, `tuning.py calibrate-checkpoint --checkpoint ...
--variant ... --data-dir ... --output fresh_calibration --recall-targets 0.5 0.6
0.7` creates a new validation selection without retraining. Use its `best.json`
for test evaluation with this version. Legacy architectures are inferred from
checkpoint metadata; use the original code if a checkpoint is incompatible.

Counts use sampled patch instances, including overlaps; they are not deduplicated
scene-level counts or false alarms per square kilometre. Coastal FP retains the
existing definition: normalized distance to shore ≤ 0.1, nominally 5 km with the
50 km normalization scale. Coastal FP at fixed overall recall still does not
prove unchanged **coastal-only recall**, which is not measured separately here.
