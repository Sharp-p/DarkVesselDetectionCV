import torch
import json

from globals import DEVICE, NUM_EPOCHS, RESULTS_DIR
from data import get_dataloaders
from network import DarkVesselNet
from evaluation import run_ablation

print("CUDA available:", torch.cuda.is_available())
print("Using device:", DEVICE)

def model_factory(fusion_mode):
    return DarkVesselNet(fusion_mode=fusion_mode)

train_loader, val_loader, test_loader = get_dataloaders()

results, table = run_ablation(
    model_factory,
    train_loader, val_loader, test_loader,
    variants=("sar_only", "early_fusion", "cgcam"),
    num_epochs=NUM_EPOCHS,
    device=DEVICE,
    results_dir=RESULTS_DIR,
)

print("\n=== Ablation table (vessel channel) ===")
for row in table:
    print(f"{row['fusion_mode']:>14} | "
          f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} | "
          f"FP={row['fp']} near-shore FP rate={row['near_shore_fp_rate']:.3f}")
