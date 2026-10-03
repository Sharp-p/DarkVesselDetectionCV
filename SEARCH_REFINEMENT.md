# Search refinement (v2): evidence from the v1 Optuna database

Source: `optuna.db` as of 2026-10-03, i.e. 59 completed real-data trials in `full-v1` (sar_only × 5 optimizers, early_fusion × AdamW) plus 20 in `pilot-v2` (sar_only/cgcam × AdamW/Lion). All used 30 epochs, batch 16, cosine schedule, vessel-F1, **one seed (42)** and fast (nondeterministic) cuDNN. Reproduce every number below with:

```bash
python analyze_optuna_db.py tuning_runs/optuna.db
```

## 1. Single-seed results have ~±0.02 F1 noise

`pilot-v2` and `full-v1` have the **same code hash, data, config, seed and parameters** for sar_only AdamW and Lion, so they are accidental replicates:

| Config (sar_only) | pilot-v2 | full-v1 | Spread |
| --- | --- | --- | --- |
| AdamW lr 1.0e-4, wd 1e-2 | 0.918 | 0.886 | 0.032 |
| AdamW lr 1.5e-4, wd 1e-2 | 0.941 | 0.947 | 0.006 |
| AdamW lr 2.9e-4, wd 1e-1 | 0.927 | 0.938 | 0.010 |
| AdamW lr 8.2e-5, wd 1e-3 | 0.921 | 0.922 | 0.002 |
| AdamW lr 1.0e-4, wd 1e-3 | 0.918 | 0.898 | 0.020 |
| Lion lr 3.0e-5, wd 0.03 | 0.374 | 0.729 | 0.355 |
| Lion lr 4.8e-5, wd 0.03 | 0.864 | 0.379 | 0.486 |
| Lion lr 2.5e-5, wd 0 | 0.409 | 0.890 | 0.481 |

The best sar_only result per optimizer was AdamW 0.947, RMSprop 0.942, SGD 0.939, Adam 0.937 and Lion 0.920. The first four are all within the replicate spread, **so v1 cannot rank AdamW, Adam, SGD and RMSprop**. v2 therefore:

- averages **2 seeds per trial by default** (`--seeds 42 43`; use 3 for a final run);
- adds `tuning.py compare`, which retrains each optimizer's selected configuration on the **same fresh seeds** and reports paired differences. Use this before claiming a ranking.

## 2. Learning-rate bounds were binding

| Optimizer | v1 range | Evidence | v2 range |
| --- | --- | --- | --- |
| AdamW | 3e-5 – 3e-4 | lr ≤ 4e-5 → 0.55–0.66 (underfit in 30 epochs); 5.7e-5 → 0.926; best 1.5e-4–2.9e-4; CGCAM's best (0.955) at 2.87e-4, on the ceiling | **5e-5 – 1e-3** |
| Adam | 3e-5 – 3e-4 | 3.2e-5 → 0.65; 5e-5–6e-5 → 0.88–0.91; flat 0.93–0.94 from 1e-4 to 2.9e-4 | **5e-5 – 1e-3** |
| SGD (Nesterov) | 1e-3 – 1e-2 | 1.2e-3–2.6e-3 → 0.90–0.92; best 5.5e-3–9.9e-3, on the ceiling | **3e-3 – 5e-2** |
| RMSprop | 1e-4 – 1e-3 | flat 0.91–0.94 everywhere; best 1.34e-4 near the floor | **5e-5 – 1e-3** |
| Lion | 1e-5 – 1e-4 | all 5 runs with lr ≤ 2.2e-5 peaked by epoch 1–6 and never improved; only consistent runs at 9.5e-5–9.9e-5, on the ceiling | **5e-5 – 1e-3** |

The v1 Lion range followed the usual "3–10× smaller than AdamW" rule. With the small number of optimizer steps here (30 short epochs), sign updates of 1e-5 move each weight too little in total. Lion's bimodality (§1) means a single Lion trial is close to a coin flip, so it especially needs multiple seeds.

## 3. Weight decay showed no measurable effect

No optimizer showed a weight-decay effect larger than the replicate noise (AdamW, for example: 0.933 at wd 0 vs 0.947 at wd 1e-2, at the same lr). The categorical choices are trimmed so trials go to the learning rate, which clearly matters:

| Optimizer | v1 choices | v2 choices |
| --- | --- | --- |
| AdamW (decoupled) | 0, 1e-4, 1e-3, 1e-2, 1e-1 | 0, 1e-2, 1e-1, 3e-1 |
| SGD (L2) | 0, 1e-5, 1e-4, 1e-3 | 0, 1e-4, 1e-3 |
| RMSprop (L2) | 0, 1e-5, 1e-4, 1e-3 | 0, 1e-4, 1e-3 |
| Lion (decoupled) | 0, 1e-3, 1e-2, 0.03, 0.1, 0.3 | 0, 1e-3, 1e-2, 1e-1 |

## 4. Pruning would have discarded good runs

Several eventual 0.89–0.93 runs were far behind at the usual pruning checkpoints: AdamW 5.7e-5 was at 0.34 at epoch 10 and finished at 0.926, and Lion 2.5e-5 was still at 0.43 at epoch 15 and finished at 0.890. SGD is at 0.75–0.91 by epoch 5 while Adam(W) is at 0.15–0.79, so the optimizers have different learning-curve shapes. A median pruner in a mixed-optimizer study would therefore bias against the Adam family. Pruning stays **off**. If enabled, it waits for the balanced phase and a full first seed.

Best epochs were 8–30 (Adam(W) mostly 19–30), so 30 epochs is kept. Changing the budget would also change what "best learning rate" means.

## 5. One study per variant, optimizer as a hyperparameter

v2 creates `<prefix>__<variant>` with `optimizer` as a categorical parameter. Learning rate and decay are conditional parameters (`adamw_lr`, `lion_weight_decay`, …), modelled by TPE with `multivariate=True, group=True`. Fairness controls:

1. **Warm starts**: each optimizer's best v1 configuration (rounded) is the first trial for that optimizer.
2. **Balanced phase**: round-robin until each optimizer has `--min-trials-per-optimizer` trials (default 4). Only then may TPE spend the rest of `--trials` where it looks most promising.
3. **Confirmation**: `compare` retrains each optimizer's best on shared fresh seeds, because the search maximum is biased toward optimizers that received more trials.

Old v1 trials are **not imported** into the new study. They used a different code hash, a single seed and different ranges. They inform the ranges and warm starts only.

## 6. The "Too many open files" crash

The 60th trial in one process (`full-v1__early_fusion__adamw` #9) produced no epoch for 28 minutes and then failed. That is consistent with DataLoader workers losing communication after descriptor exhaustion. Two mechanisms were addressed:

- **Confirmed leak:** when a trial raised mid-training (pruning, NaN, OOM, Ctrl-C), the exception traceback kept `fit_one`'s frame alive. With it stayed both loaders' persistent worker processes and their pipes: about 20 fds and 4 processes per interrupted trial at `--workers 2`, roughly double at `--workers 4`. `fit_one` now shuts workers down in a `finally` block. A regression test covers this.
- **Unconfirmed happy-path growth:** this could not be reproduced on CPU (fd count is flat across trials), so it may be CUDA/pin-memory specific. To cover it regardless, v2:
  - raises the soft `RLIMIT_NOFILE` to the hard limit (up to 65536) at start-up;
  - uses `torch.multiprocessing` `file_system` sharing (no fd per in-flight tensor), which is what PyTorch recommends for this error;
  - builds each split's dataset once per process instead of once per trial;
  - logs `fds=` each epoch and `open_fds_start/end` per trial, so a remaining leak is visible;
  - if fds grow past half the headroom, or a trial hits EMFILE, it marks the trial FAIL, queues an identical retry, writes all outputs and **re-executes the same command**. The study resumes from storage with a clean descriptor table.
