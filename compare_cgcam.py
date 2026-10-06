#!/usr/bin/env python3
"""Run controlled CGCAM experiments through the existing Optuna workflow.

See CGCAM_README.md. Uses train/validation only; no test evaluation is launched.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

from optimizers import OPTIMIZERS, SEARCH_SPACES


GROUPS = {
    "control": ("global", 0.0, ["sar_only", "early_fusion", "cgcam_no_wind", "cgcam"]),
    "gamma": ("global", 0.1, ["cgcam_no_wind", "cgcam"]),
    "local": ("local_gate", 0.1, ["cgcam_no_wind", "cgcam"]),
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--labels")
    p.add_argument("--output", default="cgcam_comparison")
    p.add_argument("--study-prefix", default="cgcam-comparison-v1")
    p.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    p.add_argument("--optimizer", choices=OPTIMIZERS, default="rmsprop")
    p.add_argument("--lr", type=float, default=0.0002649149454284989)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--diagnostics-every", type=int, default=10)
    p.add_argument("--grpc-host")
    p.add_argument("--grpc-port", type=int, default=13000)
    p.add_argument("--storage", help="Default: an absolute SQLite URL inside --output")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and commands without creating files/training")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("Learning rate must be positive and weight decay non-negative, both finite")
    if args.optimizer == "adam" and args.weight_decay != 0:
        p.error("Adam is the zero-weight-decay control")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        p.error("Seeds must be distinct integers in [0, 2**32)")
    if min(args.epochs, args.batch_size, args.cpu_threads) < 1 or min(args.workers, args.diagnostics_every) < 0:
        p.error("Epochs, batch size and CPU threads must be positive; workers/diagnostics non-negative")
    if len(set(args.groups)) != len(args.groups):
        p.error("Do not repeat experiment groups")
    root = Path(__file__).resolve().parent
    out = Path(args.output).resolve()
    space_path = out / "fixed_optimizer_space.json"
    space = copy.deepcopy(SEARCH_SPACES)
    space[args.optimizer] = {"lr": [args.lr, args.lr], "weight_decay": [args.weight_decay]}
    storage = args.storage or f"sqlite:///{out/'optuna.db'}"
    prefix = args.study_prefix
    if args.smoke and not prefix.startswith("smoke"):
        prefix = "smoke-" + prefix
    commands = []
    for group in args.groups:
        mode, gamma, variants = GROUPS[group]
        cmd = [sys.executable, str(root/"tuning.py"), "search",
               "--variants", *variants, "--optimizers", args.optimizer,
               "--cgcam-mode", mode, "--gamma-init", str(gamma),
               "--trials", "1", "--search-space", str(space_path),
               "--study-prefix", f"{prefix}-{group}", "--output", str(out),
               "--storage", storage, "--seeds", *map(str, args.seeds),
               "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--workers", str(args.workers), "--device", args.device,
               "--cpu-threads", str(args.cpu_threads), "--data-seed", str(args.data_seed),
               "--diagnostics-every", str(args.diagnostics_every), "--data-dir", args.data_dir]
        if args.labels:
            cmd += ["--labels", args.labels]
        if args.grpc_host:
            cmd += ["--grpc-host", args.grpc_host, "--grpc-port", str(args.grpc_port)]
        if args.smoke:
            cmd += ["--smoke"]
        commands.append(cmd)
    runs = sum(len(GROUPS[group][2]) for group in args.groups)*len(args.seeds)
    print(f"{runs} fresh training runs, {args.epochs} epochs each ({runs*args.epochs} total epochs).", flush=True)
    print(f"Fixed {args.optimizer}: lr={args.lr}, weight_decay={args.weight_decay}. No pruning. Validation only.", flush=True)
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    if space_path.exists() and json.loads(space_path.read_text()) != space:
        p.error("Optimizer settings changed; use a new --output directory")
    space_path.write_text(json.dumps(space, indent=2) + "\n")
    for cmd in commands:
        subprocess.run(cmd, check=True)
    rows, missing = [], []
    for group in args.groups:
        for variant in GROUPS[group][2]:
            path = out/f"{prefix}-{group}__{variant}__{args.optimizer}"/"best.json"
            if path.exists():
                manifest = json.loads(path.read_text())
                rows.append({"group": group, "variant": variant,
                             "cgcam_mode": manifest["config"]["cgcam_mode"],
                             "gamma_init": manifest["config"]["gamma_init"],
                             "optimizer": args.optimizer, "params": manifest["params"],
                             "validation_f1": manifest["validation_objective"],
                             "best_manifest": str(path.relative_to(out))})
            else:
                missing.append(str(path.relative_to(out)))
    report = out/f"{prefix}_comparison_{'-'.join(args.groups)}.json"
    report.write_text(json.dumps({"smoke": args.smoke, "split": "validation", "results": rows,
                                  "complete": not missing, "missing_manifests": missing}, indent=2) + "\n")
    print(f"Comparison report: {report}")
    if missing:
        raise SystemExit("Comparison incomplete: inspect failed trials in trials.json; "
                         "fix the cause and rerun with a fresh output directory/prefix.")


if __name__ == "__main__":
    main()
