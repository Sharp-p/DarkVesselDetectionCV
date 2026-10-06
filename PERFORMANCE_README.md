# CPU performance update

This update targets CPU overhead in validation, exact F1 calculation, data loading,
and diagnostic logging. The fusion architectures and optimizer choices are unchanged.
No detections are capped, confidence thresholds raised, or validation epochs skipped.

## Measured improvement

A reproducible CPU microbenchmark uses four 256x256 patches, two output channels,
58,753 candidate peaks, two PyTorch CPU threads, and the median of three timed
runs after warm-up:

| Stage | Previous archive | This update | Speedup |
| --- | ---: | ---: | ---: |
| Candidate extraction | 1.227 s | 0.152 s | 8.07x |
| Exact matching / F1 / AP / sweep | 0.174 s | 0.070 s | 2.48x |

Candidate records, selected F1 thresholds, sweep counts, and fixed-threshold
metrics matched the reference. AP matched within 1e-12 (summation order changes).
These are synthetic CPU-stage measurements, **not end-to-end training or GPU
speedups**. Actual gains depend on prediction density, CPU, disk, and GPU.
The numerical benchmark report is in `cpu_benchmark.json`.

## What was fixed

- Validation previously indexed PyTorch tensors and converted scalar values in
  Python for every peak. The update gathers sparse coordinates/scores in batches,
  uses NumPy indexing, and converts whole arrays at once. Dense score maps no
  longer need to be copied back for per-pixel Python processing.
- Detection matching uses an exact spatial index to inspect only nearby GT.
  It keeps confidence-first greedy matching, nearest-unmatched selection,
  distance boundaries, and tie breaking. Cumulative precision/recall/F1 arrays
  replace a Python dictionary per unique score. Test evaluation reuses each
  curve for frozen thresholds, threshold 0.3, and AP instead of recomputing it.
- Loader workers persist between epochs, retaining their scene memmaps. Prefetch
  depth is configurable. Worker OpenCV threading is disabled and PyTorch worker
  threads are limited to one to reduce oversubscription.
- Workers default to `spawn`, avoiding inherited CUDA/thread-pool state. Its
  startup cost is amortized by persistent workers. A configurable timeout turns
  stalled workers into an error rather than an indefinite silent wait.
- Gaussian target kernels are cached. Sorted target rows enable binary lookup
  of each patch's row range. Augmentation materializes its array only once.
- Pinned-loader batches use non-blocking transfers. Training loss conversion
  and diagnostic scalar transfers are deferred to the epoch boundary. The
  all-background focal-loss case uses a tensor branch rather than a host scalar
  condition; both loss values and gradients are tested against the reference.
- Candidate-cache gzip uses compression level 1, preserving data while favoring
  write speed over the smallest file size.

## Run it

Dependencies are unchanged; follow `TUNING_README.md`. Activate the environment
from the extracted project root, then start a **fresh** comparison:

```bash
python compare_cgcam.py \
  --data-dir /absolute/path/to/dataset \
  --output cgcam_cpu_v2 --study-prefix cgcam-cpu-v2 \
  --device cuda --workers 4 --cpu-threads 2 --prefetch-factor 2 \
  --diagnostics-every 50
```

This retains the default 24 real-data training runs described in `CGCAM_README.md`.
Use `--groups control`, `--groups gamma`, or `--groups local` to run stages
separately; keep all other arguments consistent. Add `--dry-run` to inspect the
plan. The same loader flags are available on `tuning.py search` and
`calibrate-checkpoint`, and saved settings are used by repeat/evaluate.

Do not increase worker count blindly. Compare 2 and 4 workers first. Prefetched
256x256 float32 input/target batches at batch size 16 contain about 28 MiB each;
4 workers x prefetch 2 is about 224 MiB of queued payload **per loader**, before
other copies and buffers. Training and validation can both retain worker pools.
Use a local SSD for the memory-mapped scene files where possible.

| Option | Default | Meaning |
| --- | --- | --- |
| `--workers` | Existing workflow default (comparison: 4) | Loader processes per split |
| `--prefetch-factor` | 2 | Queued batches per worker |
| `--persistent-workers` | Enabled | Reuse workers across epochs |
| `--no-persistent-workers` | Opt-in | Release workers after each pass |
| `--loader-start-method` | `spawn` | Alternatives: `fork`, `forkserver` on supported systems |
| `--loader-timeout` | 120 seconds | Worker wait limit; 0 disables the limit |
| `--cpu-threads` | 4 | Main-process PyTorch CPU threads, not worker count |
| `--diagnostics-every` | 10 | Sample diagnostics every N batches; 0 disables |

Worker-specific settings are ignored when `--workers 0`. Use that mode to
debug environments with blocked multiprocessing or very limited shared memory.
The CLI prints the selected device and whether CUDA is available. Confirm
`Device: cuda` when GPU training is intended.

## Find the remaining bottleneck

Each epoch now prints and saves these timings in `history.json`:

- `train_s`: total training epoch wall time.
- `data_wait_s`: time the main loop waits for its next batch, included in training
  time. It can overlap GPU work, so it is not an exact GPU-idle measurement.
- `val_s`: validation loading, inference and candidate extraction.
- `metrics_s`: exact matching, F1 calibration, AP and PR sweep.
- `save_s`: best-checkpoint and validation-summary writing.

High data wait suggests loader/I/O pressure. High validation or metric time
suggests prediction processing. High save time suggests storage/checkpoint I/O.
High CPU utilization alone does not prove that the GPU is being starved.
Timing logs do not require additional per-batch CUDA synchronization.

Run the standalone benchmark without a dataset:

```bash
python benchmark_cpu.py --output benchmark_current.json
# Optional: compare against an extracted previous ZIP.
python benchmark_cpu.py --baseline-dir /path/to/previous/project \
  --output benchmark_comparison.json
```

## Existing work and reproducibility

Do not swap files under a running training process. The study code-hash guard
remains intact: use new study prefixes/output directories for this update.
Old checkpoint weights remain architecture-compatible and can be calibrated
with `tuning.py calibrate-checkpoint`; retain the old archive to reproduce old
study selections exactly.

Persistent workers change worker RNG lifetime and subsequent shuffle/augmentation
streams compared with restarting workers each epoch. Runs remain repeatable
under the same new code, seeds, worker count and options; identical training
trajectories across old/new loader settings are not promised. Use the same
settings for every condition in a controlled comparison.

## Validation scope

CPU regression tests verify matching/ties/boundaries, candidate equality, cached
Gaussian targets, focal-loss gradients, diagnostic equivalence, and existing
optimizer/checkpoint behavior. Synthetic training checks cover original and
local fusion, both wind routings, checkpoint selection and timing output.
The multiworker execution test is skipped when a host prohibits Unix-domain
sockets required by PyTorch tensor transport; it remains enabled on normal
local Linux installations. No real SAR dataset or CUDA device was available
for end-to-end throughput measurements in this environment.

Implementation references:
[PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html),
[PyTorch transfer guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html),
[SciPy exact radius queries](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.query_ball_point.html).
