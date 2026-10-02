# DarkVesselNet tuning on a local Linux machine

This guide runs **Optuna studies for AdamW, Adam, SGD + Nesterov momentum, RMSprop and Lion**, searches detection thresholds using validation F1, and evaluates the selected models on test scenes only after the choices are frozen.

You do not need previous Optuna experience. Start with the synthetic smoke test, then calibrate existing checkpoints or run a small real-data study. The original training/evaluation entry points remain available; `tuning.py` is the new tuning workflow.

## 1. What the components do

### Performance update

Fast cuDNN is now the default: benchmarking is enabled and deterministic
algorithm selection is disabled. Seeds are still set. Use `--deterministic`
with `search` or `calibrate-checkpoint` to opt into the slower cuDNN settings;
`repeat` and `evaluate` inherit the saved configuration. The optional
`--fast-cudnn` flag explicitly selects the default.

Candidate extraction now finds maxima and gathers scores/coastal flags on the
model device, transferring only sparse results to the CPU. Optuna keeps
NumPy arrays in memory and creates Python tuples only when saving candidate
caches. Greedy matching uses bounded NumPy distance matrices; threshold curves
use cumulative arrays instead of a dictionary for every score. Frozen test
metrics reuse the same curves. JSON cache schemas and detection/matching rules
remain compatible.

Loaders retain workers across epochs when `--workers` is positive. Transfers to
the device use `non_blocking=True`; CUDA loaders already pin memory. The focal
loss no longer branches on a GPU scalar. Worker RNG streams now continue across
epochs, so augmentation sequences can differ from the older worker lifecycle.

Each epoch logs training, validation and metric seconds, plus candidate counts;
`history.json` also records validation patch counts. Validation timing includes
data loading, inference, loss and candidate extraction. Training timing includes
data loading and optimization. Checkpoint/JSON writes are outside these timings.
Full validation, the 0.01 score floor, exact F1 selection, AP and coastal metrics
are retained. Early models can still produce many candidates, and validation
still requires a full inference pass. GPU throughput must be measured on your
machine; this update was checked with the existing 15 CPU tests.

**Start with a new `--study-prefix`**: the code/config hash changed, so existing
studies intentionally reject mixing the old and new implementations. Existing
candidate JSON caches can still be recalibrated without inference.

| Component | Purpose |
| --- | --- |
| `tuning.py search` | Trains models with different learning rates and weight decay; records their validation scores. |
| Optuna **trial** | One optimizer configuration, trained for the requested epochs and seeds. |
| Optuna **study** | A collection of trials for one model variant and one optimizer. |
| `./optuna-server` / `optuna_server.py` | Starts the official Optuna **gRPC storage proxy**, which stores trial history in a database. |
| Optuna Dashboard | Browser interface for trial values, parameter comparisons, intermediate scores and trial states. |
| `tuning.py calibrate-checkpoint` | Finds validation thresholds for an existing checkpoint, without retraining. |
| `tuning.py calibrate` | Repeats an exact F1 search on a saved validation candidate cache, without model inference. |
| `tuning.py repeat` | Retrains a selected fixed configuration with several seeds; no new parameter search. |
| `tuning.py evaluate` | Tests saved checkpoints at their frozen validation thresholds; exports both vessel and structure metrics. |

**Naming:** there is no assumed third-party `pip install optuna-server` dependency here. The provided `optuna-server` executable is a small launcher for Optuna's official `run_grpc_proxy_server` API. `optuna-dashboard` is the separate official browser application. The server stores trial state; it does not train models or dispatch jobs. Training runs in the `tuning.py` process you start.

You can also use Optuna directly with SQLite, without running the server. Both routes are documented below.

## 2. Install dependencies

Recommended: Linux with Python **3.11 or 3.12**, a virtual environment, and an NVIDIA GPU for real training. CPU is supported for installation checks and small experiments. The commands below assume you have extracted this project and opened a terminal in its root directory.

On Ubuntu/Debian, install Python's virtual-environment support if needed:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install **one** PyTorch build:

- **CPU:** the complete command is below.
- **NVIDIA GPU:** run `nvidia-smi`, then use the Linux/Pip/Python/CUDA installation command from [PyTorch's installation selector](https://pytorch.org/get-started/locally/), matching your driver. Run that command inside this environment instead of the CPU command. Do not install a CPU wheel and expect CUDA to become available.

```bash
# CPU installation; skip this line if installing a CUDA build instead.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Shared training, tuning, server, dashboard and test dependencies.
python -m pip install -r requirements-tuning.txt
python -m pip check
python -c "import torch, optuna; print('torch:', torch.__version__, 'optuna:', optuna.__version__, 'CUDA:', torch.cuda.is_available())"
python -m pytest -q tests/test_tuning.py
```

The requirements include NumPy, pandas, SciPy, tifffile, headless OpenCV, matplotlib, Optuna **4.5.0**, Optuna Dashboard **0.19.0**, grpcio, protobuf, and pytest. Lion is implemented locally in `optimizers.py` following the published algorithm; no extra Lion package is required. Optuna is pinned because its gRPC API is experimental. Use the same environment for server and workers.

There is no need for Jupyter, Node.js, Docker, an Optuna account, or a database administrator for the default local SQLite setup. Figures render without a desktop display. For a record of the exact environment you use:

```bash
python -m pip freeze > tuning_environment.txt
```

In every new terminal, first `cd` to the project and run `source .venv/bin/activate`.

## 3. Start the server and dashboard

In **terminal 1**, from the project root:

```bash
source .venv/bin/activate
./optuna-server --storage sqlite:///tuning_runs/optuna.db --dashboard
```

If the archive extractor did not preserve executable permissions, use:

```bash
python optuna_server.py --storage sqlite:///tuning_runs/optuna.db --dashboard
```

Leave that terminal running. Open **http://127.0.0.1:8080** in your browser. The gRPC storage service listens on **127.0.0.1:13000**. An empty dashboard is normal until a study is created. These two ports serve different purposes; do not point a web browser at port 13000.

If you prefer separate server/dashboard terminals:

```bash
python optuna_server.py --storage sqlite:///tuning_runs/optuna.db
# In another activated terminal, from the SAME directory:
optuna-dashboard sqlite:///tuning_runs/optuna.db --host 127.0.0.1 --port 8080
```

SQLite paths are relative to the working directory: running commands from different directories can silently create different databases. An absolute SQLAlchemy SQLite URL uses **four** slashes, for example `sqlite:////home/alice/DarkVesselDetectionCV/tuning_runs/optuna.db`.

The services bind to localhost by default and do not add authentication. Keep them local. Do not delete studies through the dashboard while the gRPC service is running: the Optuna 4.5 gRPC/SQLite combination has a documented deletion-cache caveat.

## 4. Run a synthetic smoke test first

In **terminal 2**:

```bash
source .venv/bin/activate
python tuning.py search \
  --smoke --study-prefix smoke-install \
  --variants cgcam --optimizers adamw lion \
  --trials 1 --epochs 2 --seeds 42 \
  --device cpu --workers 0 \
  --grpc-host 127.0.0.1
```

This runs one trial for AdamW and one for Lion on tiny **synthetic 32×32 patches**. It checks model training, checkpoint writing, numerical PR/F1 evaluation, Optuna storage and plots. The dataset is not needed. Scores from `--smoke` are installation checks, **never project results**. Smoke study prefixes must start with `smoke` so they cannot be mistaken for real runs.

In the dashboard, inspect `smoke-install__cgcam__adamw` and `smoke-install__cgcam__lion`. A trial may take a few moments before it appears complete. The trial value is higher-is-better validation F1 by default.

To test the full output path on synthetic data:

```bash
python tuning.py evaluate \
  --best tuning_runs/smoke-install__cgcam__lion/best.json \
  --output tuning_runs/smoke-install-test \
  --device cpu --confirm-frozen
```

This test command does not need the server. The output directory must be new or empty to avoid overwriting earlier reports.

**Without a server:** omit `--grpc-host`. `search` then uses `--storage sqlite:///tuning_runs/optuna.db` directly, and you may still open the same database with `optuna-dashboard`. Do not run two workers on the same study.

## 5. Prepare the real dataset

Download/unpack the project's xView3 subset separately; data and trained weights are not included in this code archive. Arrange files like this:

```text
/path/to/dataset/
  labels.csv
  05bc615a9b0e1159t/
    VH_dB.tif
    VV_dB.tif
    bathymetry.tif
    owiMask.tif
    owiWindSpeed.tif
  ...one directory for each remaining scene...
```

`labels.csv` needs the original columns used by `data.py`, including `scene_id`, `detect_scene_row`, `detect_scene_column` and `is_vessel`. TIFFs must be compatible with the existing `tifffile.memmap` reader; compressed or tiled TIFFs may need preprocessing. The tuning addition does not download, repair or reproject data.

The split in `globals.py` is:

| Split | Scene IDs |
| --- | --- |
| Train | `05bc615a9b0e1159t`, `2899cfb18883251bt`, `72dba3e82f782f67t`, `cbe4ad26fe73f118t` |
| Validation | `e98ca5aba8849b06t` |
| Test | `590dd08f71056cacv`, `b1844cde847a3942v` |

Search constructs only train and validation loaders and checks their raster manifests. Test scenes are loaded only by `evaluate`. The shared labels CSV can contain all splits, but dataset instances filter rows to their selected scene IDs.

Use `--data-dir /path/to/dataset`; use `--labels /path/to/labels.csv` only if the CSV is elsewhere.

## 6. Calibrate your existing checkpoints before retraining

This addresses the existing threshold-0.3 comparison using your saved model weights:

```bash
python tuning.py calibrate-checkpoint \
  --checkpoint results/checkpoints/best_cgcam.pt \
  --variant cgcam --seed 42 \
  --data-dir /path/to/dataset \
  --device cuda --batch-size 16 --workers 4 \
  --output tuning_runs/calibrated-cgcam
```

Use `--device cpu` if needed. Repeat with the correct checkpoint and variant for `sar_only`, `early_fusion` or `cgcam_no_wind`. The checkpoint must be a trusted project file containing `model_state_dict`. This command copies it into the output and selects thresholds on validation only; no optimizer or model weights are changed. `--seed` records the original training seed; it does not retrain the model.

The result is `best.json`, `selection.json`, `best_validation.json`, and a compressed sparse-candidate cache. The normal `evaluate` command accepts this `best.json`. `repeat` does not: legacy checkpoints do not establish a selected optimizer configuration.

You can repeat the threshold search from the cache without another forward pass:

```bash
python tuning.py calibrate \
  --cache tuning_runs/calibrated-cgcam/validation_candidates.json.gz \
  --threshold-min 0.02 --threshold-max 0.8 \
  --output tuning_runs/calibrated-cgcam/threshold-comparison.json
```

This writes a standalone comparison and deliberately does not silently replace the frozen `selection.json`. If you decide to adopt a new interval, rerun `calibrate-checkpoint` into a new output directory with that interval. The command rejects test caches. A cache cannot recover detections below its original `--min-score`; lower that floor and regenerate candidates if needed.

## 7. Start a small real-data Optuna study

A practical first run compares AdamW and Lion for radar-only and CGCAM:

```bash
python tuning.py search \
  --study-prefix pilot-v1 \
  --variants sar_only cgcam \
  --optimizers adamw lion \
  --trials 5 --epochs 30 --seeds 42 \
  --data-dir /path/to/dataset \
  --device cuda --batch-size 16 --workers 4 \
  --objective vessel-f1 \
  --grpc-host 127.0.0.1
```

That is **4 studies × 5 trials = 20 training runs**, not five runs total. At 30 epochs it is 600 training epochs. One study is named, for example, `pilot-v1__cgcam__lion`. The studies get separate budgets, preventing one optimizer from consuming the others' trial allocations.

To include **all four variants and all five optimizers**:

```bash
python tuning.py search \
  --study-prefix full-v1 \
  --variants sar_only early_fusion cgcam_no_wind cgcam \
  --optimizers adamw adam sgd rmsprop lion \
  --trials 10 --epochs 30 --seeds 42 \
  --data-dir /path/to/dataset \
  --device cuda --batch-size 16 --workers 4 \
  --grpc-host 127.0.0.1
```

This is **20 studies × 10 trials = 200 training runs**. Start smaller if time or GPU resources are limited. Every trial starts a freshly initialized model; checkpoints are not carried between optimizer trials.

For a real-data installation check, use `--max-train-samples 32 --max-val-samples 32 --epochs 2` with a **new prefix**. Subsets are selected deterministically. Do not report subset checks as full-split results. `macro-f1` requires examples of both classes in validation.

### Search parameters

| Optimizer | Learning-rate search (log scale) | Weight-decay choices | Other fixed settings |
| --- | --- | --- | --- |
| AdamW | `3e-5` to `3e-4` | `0, 1e-4, 1e-3, 1e-2, 1e-1` | betas `(0.9, 0.999)` |
| Adam control | `3e-5` to `3e-4` | `0` | betas `(0.9, 0.999)` |
| SGD | `1e-3` to `1e-2` | `0, 1e-5, 1e-4, 1e-3` | momentum `0.9`, Nesterov enabled |
| RMSprop | `1e-4` to `1e-3` | `0, 1e-5, 1e-4, 1e-3` | alpha `0.99`, eps `1e-8`, momentum `0`, uncentered |
| Lion | `1e-5` to `1e-4` | `0, 1e-3, 1e-2, 0.03, 0.1, 0.3` | betas `(0.9, 0.99)` |

These ranges are hypotheses, not established best values. The initial trial uses the documented reference settings if they are inside the chosen domain; Lion queues both `(3e-5, 0.03)` and `(1e-5, 0.1)`. With a budget of one trial, only the first runs. TPE uses five startup trials before adaptive sampling, so a very small pilot is mostly a screening experiment.

Learning rate and decay are searched jointly. Decay is categorical so **zero is a genuine candidate**, rather than an invalid point in a log distribution. Edit a copy of `tuning_search_space.json` and supply `--search-space my_search_space.json` to change domains. Keep all five optimizer entries. For example, replace the AdamW weight-decay array with `[0.001, 0.003, 0.01, 0.03]` for a focused follow-up. Use a new study prefix when changing the space.

Adam is intentionally a no-decay control; it overlaps AdamW when AdamW selects zero decay. SGD/RMSprop use PyTorch's coupled L2-style weight decay; AdamW/Lion use decoupled decay. Their coefficients are not assumed interchangeable.

For all optimizers, only matrix/kernel weights receive decay. Biases, BatchNorm affine parameters and CGCAM's scalar `gamma` are excluded. This policy differs from the legacy ablation script's decay on every parameter; record it when comparing old results. Focal-loss exponents remain `(2, 4)`, Gaussian sigma remains 2 pixels, and the matching radius remains 20 pixels. Those are not silently tuned.

The default scheduler is cosine annealing over the entire trial. Use `--scheduler none` for a constant LR. `--max-grad-norm 1.0` preserves the original gradient clipping. Batch size, scheduler, epochs and augmentation are fixed within a study, not searched. Keep these comparable between optimizers. The smaller Lion LR scales the entire cosine schedule, not only its first step.

## 8. How validation F1 and AP are computed

Each epoch performs one validation forward pass per batch. It collects **local maxima of raw logits**, then uses sigmoid scores for thresholds. This avoids artificial plateaus caused by clamping probabilities to 0.9999. The training output and loss still use the original clamped heatmaps. Boolean peak masks preserve valid zero logits. Actual equal-logit plateaus are not spatially merged; matching counts duplicate detections as false positives.

Candidates are cached sparsely: coordinates, scores, per-class GT and a near-shore flag. Whole-scene or full-batch heatmaps are not retained. There is no top-K cap. Very low score floors can still produce many candidates and require considerable CPU time/RAM.

- `--min-score 0.01`: discard lower-scoring candidate peaks before caching.
- `--threshold-min 0.01 --threshold-max 0.95`: admissible operating-point interval.
- The search examines **every distinct retained score in that interval and its endpoints**, not only the 99 plotting thresholds. This is an exact F1 search for the cached detections and the specified matching rule. No additional training is needed per threshold.
- Each prediction greedily claims its nearest unmatched GT within 20 pixels, processing higher scores first. Equal scores use a stable row/column order within a patch; global PR updates group tied scores together.
- Each class gets its own validation-selected threshold. Equal F1 prefers fewer false positives, then the higher threshold.
- The objective `vessel-f1` maximizes vessel best-F1; `macro-f1` averages the separately optimized vessel and structure F1; `vessel-ap` selects by vessel AP. The chosen checkpoint is the best epoch for that objective, not necessarily the lowest-loss epoch.
- All objectives still save both classes, exact best F1 cutoffs, fixed-0.3 results, and numerical PR sweeps. The loss plot's dashed line is explicitly the minimum validation-loss epoch; the selected objective epoch is in `selection.json`.

Optuna searches the optimizer parameters; F1 threshold selection is a deterministic inner search. Treating thresholds as additional random Optuna parameters would spend trials rediscovering a curve that is already available from one validation pass.

`ap_at_floor` is **non-interpolated detection AP**: sum of recall increments multiplied by precision at each distinct retained score. It is computed from detections, not JPEG pixels or a trapezoidal upper envelope. Scores below the candidate floor are missing, so it is explicitly a truncated AP; changing the floor changes the metric definition. Use the same floor in comparisons. A class with no GT has AP `null`, not a misleading perfect score.

**Evaluation scope matters:** the existing dataset builds target-centered and random background patches; patches may overlap and count the same scene object multiple times. These are **sampled-patch metrics**, not official full-scene xView3 benchmark scores. This addition preserves that sampling strategy. It does not implement scene stitching, scene-wide deduplication, official benchmark matching, or AIS association.

The existing data pipeline also maps missing `is_vessel` values to vessel, samples random background rather than explicit hard negatives, and approximates context alignment by resized windows. These behaviors are unchanged. Decide any corrections before the final experiment and use a new study prefix after code changes.

## 9. Resume, pruning and reproducibility

Rerun the identical search command to resume its studies. `--trials` is the **total terminal-trial budget per study**, including COMPLETE, PRUNED and FAIL states. Increase it from 5 to 10 to add up to five more trials, rather than starting ten additional ones. Failed/pruned trials consume budget; inspect them rather than silently ignoring them.

Optuna trial history is persistent in the database. Each trial stores checkpoints separately. The resume contract includes code hash, selected scenes, label hash, train/validation TIFF size/mtime manifests, optimizer space, seeds, scheduler and score definitions. A changed contract is rejected; use a new prefix instead of mixing incomparable results. Keep the same artifact directory when continuing a study. TIFF content is not fully hashed; do not replace rasters in place while preserving their metadata.

**Resuming a study is not resuming halfway through an epoch.** Ctrl-C marks an interrupted trial failed. After a reboot/SIGKILL, the next local worker marks leftover RUNNING trials failed after acquiring the study's OS lock. Increase the trial budget if you want a replacement trial. Completed trials are retained. A trial's model weights are initialized anew; the saved optimizer state is for inspection/future extension, not automatic mid-trial recovery.

One local worker per study is supported. A filesystem lock prevents accidental duplicate workers using its artifact directory. Parallelize *different* studies on separate GPUs rather than two workers on one study. Example:

```bash
# Terminal 2, GPU 0
CUDA_VISIBLE_DEVICES=0 python tuning.py search --study-prefix full-v1 \
  --variants cgcam --optimizers adamw --trials 10 --epochs 30 \
  --data-dir /path/to/dataset --device cuda --grpc-host 127.0.0.1

# Terminal 3, GPU 1
CUDA_VISIBLE_DEVICES=1 python tuning.py search --study-prefix full-v1 \
  --variants cgcam --optimizers lion --trials 10 --epochs 30 \
  --data-dir /path/to/dataset --device cuda --grpc-host 127.0.0.1
```

The per-study `best.json` and `trials.json` are authoritative. The prefix-level summary only lists the studies handled by that invocation and may be replaced by another invocation using the same prefix.

Pruning is **off by default** so optimizers receive the same full epoch budget. Add `--prune` only for exploratory screening; the median pruner waits for five completed trials and ten intermediate steps. With multiple seeds, a step is one epoch within the sequential seed runs. Pruned trials may have partial checkpoints, but cannot become `best.json`. Since optimizers can learn at different speeds, confirm promising settings without pruning in a new study.

Training seeds initialize Python, NumPy and Torch. Background sample coordinates use a separate fixed `--data-seed`; loader shuffle and workers have explicitly seeded generators. Every optimizer sees the same sampling protocol. This fixes the earlier unseeded `RandomState(None)` behavior. GPU kernels and package/device differences can still prevent bitwise equality. Optuna resumes trial history, not the exact sampler RNG stream of an uninterrupted process; fixed trial parameters and seeds are the unit of reproducibility.

## 10. Repeat finalists with several seeds

Choose candidates using validation results. Then repeat the selected configuration, for example:

```bash
python tuning.py repeat \
  --best tuning_runs/pilot-v1__cgcam__lion/best.json \
  --seeds 43 44 45 \
  --device cuda \
  --output tuning_runs/final-cgcam-lion
```

This fixes LR and decay and retrains with fresh seeds. Each seed chooses its checkpoint and thresholds using validation. It writes a new `best.json` listing **all** repeated runs, not just the most favorable seed. The original dataset paths and budget are retained. Compare all reported seeds; do not pick one based on test performance.

Alternatively, pass `--seeds 42 43 44` during `search`. Every trial then trains three models and Optuna maximizes their mean validation objective. This is more expensive. Standard deviation is the **sample** standard deviation (`n-1`); a single run reports `null`, not zero.

With only one validation scene, large searches can overfit that scene. Keep the search budget proportionate, use matched budgets across variants, and report this limitation.

## 11. Evaluate frozen selections on test data

After selecting optimizer, architecture, checkpoint-selection rule and threshold interval using validation:

```bash
python tuning.py evaluate \
  --best tuning_runs/final-cgcam-lion/best.json \
  --device cuda \
  --output tuning_runs/test-final-cgcam-lion \
  --confirm-frozen
```

`--confirm-frozen` is an explicit protocol flag, not another training step. The command:

1. Loads every listed checkpoint and verifies it matches its validation selection.
2. Loads test scenes only now.
3. Reports both classes at the saved per-class thresholds and at 0.3.
4. Saves numerical PR curves and truncated AP, but **does not choose a best test F1 threshold**.
5. Reports each seed plus mean/sample standard deviation across seeds.

The PR figure's star marks the **validation-selected threshold**, not the best point on the test curve. The numerical report has `threshold_source: validation`. The `calibrate` command refuses to optimize on a test cache.

Coastal outputs include the number of unmatched detections within 5 km of shore and `near_shore_fp_fraction_of_fp`. That fraction means coastal FP divided by all FP; it is **not false alarms per square kilometre**. Compare coastal false positives alongside recall. Raw reductions at different recalls do not demonstrate selective coastal suppression. Numerical sweeps allow inspection at comparable recall; report any post-hoc test comparison as descriptive, not a new threshold-selection procedure.

## 12. Where the files go

For a study, outputs look like:

```text
tuning_runs/pilot-v1__cgcam__lion/
  contract.json
  trials.json
  best.json
  trial_00000/
    result.json
    seed_42/
      best.pt
      selection.json
      history.json
      best_validation.json
      validation_candidates.json.gz
      loss_curves.png
      validation_pr_vessel.png
      validation_pr_structure.png
```

The best manifest uses relative paths to seed selections, so an entire study directory can be archived together. Search continuation also relies on absolute artifact paths stored in Optuna, so keep directories in place while the study is active. Code/data compatibility checks intentionally reject resuming against changed source or data; preserve the project version with the study.

Final evaluation adds `summary.json`, `metrics.csv`, and per-seed `test_metrics.json`, `test_candidates.json.gz`, and two test PR figures. These are the numerical sources to use in slides. Checkpoints can be large because they contain optimizer state. Keep the database **and** trial artifacts; the database alone cannot recover trained weights.

## 13. Troubleshooting

| Symptom | Action |
| --- | --- |
| `ModuleNotFoundError` | Activate `.venv`; rerun `python -m pip install -r requirements-tuning.txt`. |
| CUDA unavailable | Check `nvidia-smi` and that the installed Torch wheel is a CUDA build. Use `--device cpu` for smoke checks. |
| CUDA out of memory | Reduce batch size and use a **new study prefix**. Do not change batch size silently within a comparison. OOM trials are marked FAIL and the study can continue. |
| Missing scene or `labels.csv` | Check `--data-dir`, `--labels`, scene directory names and required rasters. |
| TIFF cannot be memory mapped | Check the source files' encoding. The original data reader requires a memmappable TIFF layout. |
| Worker process crash | Try `--workers 0` with a new prefix. The CLI has an import-safe main guard. |
| gRPC connection timeout | Start `optuna_server.py`; check host/port and that all terminals use the same environment. |
| Empty dashboard | Verify its database URL and working directory match the server's, and a search has started. |
| `Address already in use` | Stop the other service or change `--port` / `--dashboard-port`; set the worker's `--grpc-port` accordingly. |
| `database is locked` | Use one local worker at a time initially. SQLite has a 60-second timeout; for sustained concurrency use PostgreSQL. |
| Contract mismatch | Keep the previous study intact and choose a new prefix for changed code, seeds, data, settings or search ranges. |
| Search reports no additional trials | The total terminal-trial budget has already been reached; increase `--trials`. |
| No vessel/structure GT | Remove/reconsider tiny validation subsets; `macro-f1` needs both classes. |
| F1 threshold lies at a search boundary | Inspect the validation curve. Expand the interval and, if needed, lower the cache floor in a new experiment. |
| Single-run standard deviation is null | Expected: one run cannot estimate variability. Use `repeat`. |

For PostgreSQL, provision a database/user locally, install `psycopg2-binary`, and give the server/dashboard a SQLAlchemy URL such as `postgresql+psycopg2://USER:PASSWORD@127.0.0.1/optuna`. Workers can keep using gRPC. SQLite is sufficient for the sequential local workflow; this project does not implement cross-machine artifact synchronization or multi-worker coordination within one study.

## 14. References and implementation notes

- [Optuna gRPC server API (4.5)](https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.storages.run_grpc_proxy_server.html)
- [Optuna gRPC client API (4.5)](https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.storages.GrpcStorageProxy.html)
- [Optuna Dashboard setup](https://optuna-dashboard.readthedocs.io/en/stable/getting-started.html)
- [Lion paper](https://arxiv.org/abs/2302.06675) and [authors' tuning guidance](https://github.com/google/automl/tree/master/lion)
- [AdamW paper](https://arxiv.org/abs/1711.05101)

The tests cover Lion updates/state loading, all optimizer factories, decay exclusions, exact F1/AP math including tied scores, agreement with independent greedy matching, logit peak extraction, deterministic background sampling, and validation/test separation. Synthetic end-to-end runs exercise the CLI, database/server and artifact paths. They do not establish performance on real SAR data.
