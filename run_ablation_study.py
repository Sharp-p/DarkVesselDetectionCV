"""
run_ablation_study.py

Final evaluation after the Optuna search: for each fusion variant, retrain the winning
optimizer/hyperparameters with fresh seeds, pick the epoch and the confidence threshold
on validation, score the test split once, and report mean +/- std over seeds.

    QUICK=1 python run_ablation_study.py   # 1 seed, 2 epochs: checks the pipeline end to end
    python run_ablation_study.py           # real run
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import optuna

from evaluation import (NEAR_SHORE_KM_THRESHOLD, SHORE_CHANNEL_INDEX, plot_pr_curves,
                        plot_prediction_overlays, run_ablation)
from globals import DEVICE, MAX_DISTANCE_SHORE_METERS, NUM_EPOCHS, RESULTS_DIR
from network import DarkVesselNet
from tuning_runtime import make_loader
from utils import load_model_for_inference, predict

PREFIX = "full-v2"
STORAGE = "sqlite:///tuning_runs/optuna.db"
OPTIMIZERS = ["adamw", "adam", "sgd", "rmsprop", "lion"]
VARIANTS = ["sar_only", "early_fusion", "cgcam_no_wind", "cgcam"]

QUICK = os.environ.get("QUICK") == "1"
SEEDS = [1] if QUICK else [1, 2, 3]   
EPOCHS = 2 if QUICK else NUM_EPOCHS
OUT_DIR = RESULTS_DIR + ("_quick" if QUICK else "")


def best_configs(variants):
    """Per variant, the best optimizer and hyperparameters by validation score."""
    configs = {}
    for v in variants:
        best = None
        for o in OPTIMIZERS:
            study = optuna.load_study(study_name=f"{PREFIX}__{v}__{o}", storage=STORAGE)
            if best is None or study.best_value > best[0]:
                best = (study.best_value, o, study.best_params)
        score, optimizer, params = best
        configs[v] = {"optimizer": optimizer, "lr": params["lr"],
                      "weight_decay": params["weight_decay"]}
        print(f"{v}: {optimizer} {configs[v]} (search val F1 {score:.4f})")
    return configs


def model_factory(fusion_mode):
    return DarkVesselNet(fusion_mode=fusion_mode)


optimizer_configs = best_configs(VARIANTS)

# Same data pipeline and training settings as the search.
winner = optimizer_configs["cgcam"]["optimizer"]
selection_path = glob.glob(f"tuning_runs/{PREFIX}__cgcam__{winner}/**/selection.json",
                           recursive=True)[0]
cfg = json.load(open(selection_path))["config"]
if not QUICK:
    assert cfg["epochs"] == NUM_EPOCHS, (cfg["epochs"], NUM_EPOCHS)

# ---- train and evaluate each seed ----
all_results, all_tables = {}, {}
for seed in SEEDS:
    all_results[seed], all_tables[seed] = run_ablation(
        model_factory,
        lambda fm, s=seed: make_loader(cfg, "train", s),
        lambda fm: make_loader(cfg, "val", cfg["data_seed"]),
        lambda fm: make_loader(cfg, "test", cfg["data_seed"]),
        variants=tuple(VARIANTS),
        optimizer_configs=optimizer_configs,
        num_epochs=EPOCHS,
        device=DEVICE,
        results_dir=os.path.join(OUT_DIR, f"seed_{seed}"),
        seed=seed,
        max_grad_norm=cfg["max_grad_norm"],
    )
results = all_results[SEEDS[0]]   # curves and overlays below use the first seed

# ---- aggregate over seeds ----
keys = ("precision", "recall", "f1", "fp", "near_shore_fp_rate")
agg = {}
print(f"\n=== Ablation table (vessel channel), mean +/- std over seeds {SEEDS} ===")
for v in VARIANTS:
    rows = [next(r for r in all_tables[s] if r["fusion_mode"] == v) for s in SEEDS]
    agg[v] = {k: (float(np.mean([r[k] for r in rows])),
                  float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0)
              for k in keys}
    m = agg[v]
    print(f"{v:>14} | P={m['precision'][0]:.3f}±{m['precision'][1]:.3f} "
          f"R={m['recall'][0]:.3f}±{m['recall'][1]:.3f} "
          f"F1={m['f1'][0]:.3f}±{m['f1'][1]:.3f} | "
          f"FP={m['fp'][0]:.0f}±{m['fp'][1]:.0f} "
          f"near-shore FP rate={m['near_shore_fp_rate'][0]:.3f}±{m['near_shore_fp_rate'][1]:.3f}")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "ablation_aggregate.json"), "w") as f:
    json.dump({"seeds": SEEDS, "epochs": EPOCHS, "configs": optimizer_configs,
               "variants": agg}, f, indent=2)

# ---- figures ----
plot_pr_curves({fm: r["sweep"] for fm, r in results.items()}, channel=1,
               save_path=os.path.join(OUT_DIR, "pr_structure_comparison.png"),
               title="Precision-recall comparison: Fixed structure detection")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(VARIANTS,
       [agg[v]["near_shore_fp_rate"][0] for v in VARIANTS],
       yerr=[agg[v]["near_shore_fp_rate"][1] for v in VARIANTS],
       capsize=4, color=["#999", "#999", "#999", "#2a7"])
ax.set_ylabel("Near-shore FP rate (vessel channel)")
ax.set_title(f"Fraction of false positives within {NEAR_SHORE_KM_THRESHOLD:g}km of shore")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "near_shore_fp_comparison.png"), dpi=200)
plt.close(fig)

# ---- near-shore overlays: a few near-shore test patches, reused for each variant ----
near_shore_batches = []
for inputs, targets in make_loader(cfg, "test", cfg["data_seed"]):
    shore_km = inputs[:, SHORE_CHANNEL_INDEX].mean(dim=(1, 2)) * (MAX_DISTANCE_SHORE_METERS / 1000)
    mask = shore_km <= NEAR_SHORE_KM_THRESHOLD
    if mask.any():
        near_shore_batches.append((inputs[mask][:3], targets[mask][:3]))
    if len(near_shore_batches) >= 2:
        break

for fusion_mode in ("sar_only", "cgcam"):
    model = model_factory(fusion_mode)
    load_model_for_inference(model, results[fusion_mode]["checkpoint_path"], device=DEVICE)
    for i, (inputs, targets) in enumerate(near_shore_batches):
        pred = predict(model, inputs, device=DEVICE)
        plot_prediction_overlays(
            inputs, pred, targets,
            save_dir=os.path.join(OUT_DIR, "overlays", fusion_mode),
            prefix=f"nearshore_batch{i}",
            confidence_threshold=results[fusion_mode]["threshold"],
        )
