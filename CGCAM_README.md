# CGCAM experiments: gamma initialization and local context gating

**Matched recall update:** [MATCHED_RECALL_README.md](MATCHED_RECALL_README.md) covers recall-constrained checkpoint selection and automatic total/coastal false-positive comparisons.

This update makes the architectural hypotheses testable. It does not assume that
zero-initialized gamma is a bug, or that changing fusion will improve accuracy.
The original global attention remains the default. The comparison launcher
explicitly selects the new settings as separate experiments.

## What changed

| Setting | Fusion | Initial gamma | Purpose |
| --- | --- | ---: | --- |
| `control` | Original global cross-attention | 0.0 | Reference experiment |
| `gamma` | Original global cross-attention | 0.1 | Isolate initialization |
| `local` | Aligned local spatial/channel gate | 0.1 | Compare fusion at the same initialization |

Both `cgcam` and `cgcam_no_wind` support all three settings. SAR-only and early
fusion are trained once per seed as controls. The four existing variant names
still describe input routing; `cgcam_mode` separately describes the fusion module.

The local alternative combines context and radar features at the same location,
uses a depthwise 3x3 convolution in latent space, and predicts a signed gate:

```python
signed_gate = 2 * sigmoid(gate_logits) - 1
output = radar + gamma * radar * signed_gate
```

With gamma initialized to 0.1, the initial feature multiplier is between 0.9 and
1.1. This permits attenuation and amplification. Gamma is trainable and is not
bounded, so these bounds apply to initialization, not every future step. The
local alternative has no dense N-by-N attention matrix. It has a different
parameter count and computation pattern; benchmark runtime rather than assuming
it is faster. BatchNorm uses batch/spatial statistics during training; “local”
describes its spatial mixing, not complete independence from other locations.

All models still use the same 32x32 bottleneck for a 256x256 input. This update
does not add higher-resolution paths or positional encoding to global attention.

CGCAM context/fusion initialization now uses an isolated CPU RNG stream. For a
given seed, shared radar and decoder weights match between SAR-only and either
CGCAM mechanism. The context encoder also starts identically for both CGCAM
mechanisms. This improves the controlled comparison, but new seeded training
will not reproduce the old archive's initialization exactly. Early fusion has a
different input stem and is not promised identical shared initialization.

## Linux installation

Use the dependency installation and server instructions in **TUNING_README.md**.
There are no additional dependencies for this update. From the extracted project
root, activate your environment and run:

```bash
source .venv/bin/activate
python -m pytest -q tests
```

## Start with an installation check

This trains eight tiny synthetic models for one epoch each. The resulting scores
are not scientific results. Output is separated from real experiments.

```bash
python compare_cgcam.py \
  --smoke --epochs 1 --seeds 42 \
  --device cpu --workers 0 --diagnostics-every 1 \
  --output smoke_cgcam
```

## Controlled real-data comparison

```bash
python compare_cgcam.py \
  --data-dir /absolute/path/to/dataset \
  --output cgcam_comparison_v1 \
  --optimizer rmsprop --lr 0.0002649149454284989 --weight-decay 0 \
  --seeds 42 43 44 --epochs 30 \
  --device cuda --batch-size 16 --workers 4
```

The default optimizer settings match the no-wind RMSprop configuration in the
provided aggregate reports. They are a fixed control, not a claim that RMSprop
is optimal for every architecture. All conditions get exactly the same optimizer,
learning rate, weight decay, cosine schedule, seeds, sampling seed, batch size,
and epoch budget. Gamma is excluded from weight decay, as before.

This launches **8 configurations x 3 seeds = 24 training runs**, each up to 30
epochs: **720 total training epochs**, with no pruning. Each configuration is one
Optuna trial whose objective averages its seed scores; it is not a 24-trial
learning-rate search. Validation selects the checkpoint and per-class detection
threshold separately for each seed. Test scenes are not loaded.

To inspect the exact commands without training, append `--dry-run`. To proceed
in stages, use the same arguments and directory but select a group:

```bash
# First: original attention plus SAR-only and early-fusion controls.
python compare_cgcam.py --data-dir /absolute/path/to/dataset \
  --output cgcam_comparison_v1 --groups control

# Second: change gamma initialization only.
python compare_cgcam.py --data-dir /absolute/path/to/dataset \
  --output cgcam_comparison_v1 --groups gamma

# Third: change fusion while holding gamma initialization at 0.1.
python compare_cgcam.py --data-dir /absolute/path/to/dataset \
  --output cgcam_comparison_v1 --groups local
```

These staged examples use the same defaults as the full example. Repeat any
custom seed/optimizer/training arguments consistently when running in stages.
Restarting an unchanged command skips completed trials. Interrupted training is
not resumed within an epoch; the existing Optuna recovery rules apply. A failed
one-trial study consumes its budget: inspect the error, fix it, and use a new
output directory/prefix to rerun it. Changed code/settings also require a fresh
study prefix; changed optimizer settings require a new comparison output folder.

SQLite is used directly by default. To use the existing server/dashboard,
start it with the database you want to inspect and append `--grpc-host 127.0.0.1`
to the comparison command. The server's database is authoritative in gRPC mode;
the launcher's local `--storage` is not used by the proxy client.

## Wind versus no wind with the same optimizer

Use `--wind-only` to exclude SAR-only and early-fusion controls. For the local
fusion pair, run:

```bash
python compare_cgcam.py --wind-only --groups local \
  --data-dir /absolute/path/to/dataset --output wind_pair_rmsprop \
  --optimizer rmsprop --lr 0.0002649149454284989 --weight-decay 0 \
  --seeds 42 43 44 --epochs 30 --device cuda --workers 4
```

This runs two variants across three seeds: six training runs, 180 total epochs.
Both variants use identical optimizer settings (including fixed betas/momentum),
learning rate, weight decay, scheduler, seeds, data split, batch size and epoch
budget. Only wind availability differs within this pair. Validation still chooses
the checkpoint and detection threshold separately for each run. Add `--dry-run`
to inspect the commands. Use `--groups control` for original attention or
`--groups gamma` for attention with gamma initialized to 0.1. Choose a fresh
output directory when changing settings. To compare another optimizer, change
`--optimizer`, `--lr`, and `--weight-decay` together for the whole pair; this
launcher does not tune the two variants independently.

## Inspect results and diagnostics

Each condition has an Optuna study named, for example:

```text
cgcam-comparison-v1-local__cgcam_no_wind__rmsprop
```

Inside it, `best.json` points to each selected seed checkpoint and selection.
The top-level `*_comparison_*.json` summarizes validation F1 mean/sample standard
deviation across seeds. Each seed's `history.json` contains `fusion_diagnostics`:

| Field | Meaning |
| --- | --- |
| `gamma_start`, `gamma_end` | Learned scalar at the beginning/end of the epoch |
| `relative_branch_norm_mean` | Mean sampled `norm(output - radar) / norm(radar)` |
| `context_grad_norm_mean` | Context-encoder gradient norm before clipping |
| `fusion_grad_norm_mean` | Fusion-weight gradient norm, excluding gamma, before clipping |
| `gamma_grad_abs_mean` | Mean absolute gradient of gamma before clipping |
| `first_sample` | First sampled step, including the initial zero-gamma behavior |

Diagnostics sample the first training batch and every tenth batch by default.
Use `--diagnostics-every 1` for every batch, or `0` to disable. They detach all
observations, retain no computation graphs, and are not collected on validation
or test passes. Norms are batch norms, not per-example averages. At gamma=0,
zero context gradients on the first step are expected; gamma may still receive a
gradient. Examine later steps/epochs and the branch contribution before declaring
the branch inactive. A small gamma alone does not prove a negligible contribution.

## Continue Optuna optimizer searches

All five optimizers and their previous search spaces remain available. The new
options are fixed per study, recorded in its contract, and restored for repeat
and evaluation. Use a different prefix for each fusion/initialization experiment:

```bash
python tuning.py search \
  --variants cgcam_no_wind cgcam --optimizers adamw rmsprop lion \
  --cgcam-mode local_gate --gamma-init 0.1 \
  --study-prefix local-gate-tuning-v1 \
  --trials 5 --epochs 30 --seeds 42 \
  --data-dir /absolute/path/to/dataset --device cuda
```

That example is 2 variants x 3 optimizers x 5 trials x 1 seed = 30 training runs.
Gamma's **initial value** is fixed by the CLI; gamma then learns with the other
parameters. Optuna searches learning rate and weight decay, not gamma's initial
value. Optimizer betas retain their previous fixed values.

## Evaluate selected checkpoints

After selecting the configuration on validation data, use its manifest:

```bash
python tuning.py evaluate \
  --best cgcam_comparison_v1/cgcam-comparison-v1-local__cgcam_no_wind__rmsprop/best.json \
  --output frozen_local_test --device cuda --confirm-frozen
```

The model architecture and learned gamma are restored automatically. The test
thresholds remain those selected on validation. For coastal comparisons, compare
false positives at comparable recall using the numerical PR data; a raw FP
reduction at lower recall does not alone establish selective suppression. A
threshold chosen retrospectively on a test curve is descriptive, not an unbiased
operating point. Keep further architectural decisions on validation data.

## Existing checkpoints and direct Python use

Global attention retains its original parameter keys and tensor shapes, so old
global checkpoints can be loaded into a global model. Local-gate weights are a
different architecture and require fresh training; do not use `strict=False` to
pretend an old attention checkpoint is a local-gate checkpoint. Existing Optuna
studies retain their code-hash guard and cannot be resumed under this changed
code. Keep the previous archive to reproduce or evaluate previous selections.

For a new checkpoint:

```python
import torch
from network import DarkVesselNet

checkpoint = torch.load("path/to/best.pt", map_location="cpu", weights_only=True)
model = DarkVesselNet(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

For fresh training, explicitly choose the new local setting:

```python
model = DarkVesselNet("cgcam_no_wind", cgcam_mode="local_gate", gamma_init=0.1)
```

The default constructor still means global attention with gamma initialized to
zero. `tuning.py calibrate-checkpoint` infers architecture from new checkpoint
metadata; legacy checkpoints without that metadata default to global attention.
It recalibrates on validation, not test. Loading weights restores **learned**
gamma, regardless of the constructor's initial value.
