#!/usr/bin/env python3
"""Optuna optimizer studies, validation-only F1 calibration, and frozen test evaluation.

See TUNING_README.md for a complete local Linux walkthrough.
"""
import argparse
from contextlib import contextmanager
import csv
import fcntl
import gc
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
import time

import optuna
import torch

import globals as config
from network import DarkVesselNet, FUSION_MODES, CGCAM_MODES
from optimizers import DEFAULTS, OPTIMIZERS, SEARCH_SPACES, make_optimizer
from optuna_server import client_storage
from train import train_one_epoch
from tuning_metrics import DetectionCurve, at_frozen_thresholds, summarize
from tuning_runtime import (code_digest, collect_candidates, dataset_fingerprint,
                            file_sha256, make_loader)

SCHEMA = 1


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_cache(path, records, cfg, split):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump({"schema": SCHEMA, "split": split, "candidate_floor": cfg["min_score"],
                   "smoke": cfg["smoke"], "records": records}, f, allow_nan=False)
    tmp.replace(path)


def load_cache(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


@contextmanager
def study_lock(directory):
    """One local worker per study; independent studies can use different GPUs."""
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / ".worker.lock", "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker is using {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def aggregate(values):
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def objective_score(summary, objective):
    if summary["gt_counts"][0] == 0:
        raise ValueError("Validation has no vessel GT; use a larger validation subset")
    if objective == "vessel-ap":
        return summary["ap_at_floor"][0]
    if objective == "macro-f1":
        if summary["gt_counts"][1] == 0:
            raise ValueError("macro-f1 requires structure GT as well as vessel GT")
        return statistics.mean(m["f1"] for m in summary["best_by_channel"])
    return summary["best_by_channel"][0]["f1"]


def summarize_cfg(records, cfg):
    return summarize(records, min_score=cfg["min_score"],
                     threshold_min=cfg["threshold_min"], threshold_max=cfg["threshold_max"],
                     distance_threshold=cfg["match_distance"])


def build_model(cfg):
    return DarkVesselNet(fusion_mode=cfg["variant"],
                         cgcam_mode=cfg.get("cgcam_mode", "global"),
                         gamma_init=cfg.get("gamma_init", 0.0)).to(cfg["device"])


def fit_one(cfg, optimizer_name, params, seed, directory, report=None):
    """Train and choose epoch + per-class cutoffs entirely on validation data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config.set_seed(seed)
    model = build_model(cfg)
    optimizer = make_optimizer(model, optimizer_name, **params)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
                 if cfg["scheduler"] == "cosine" else None)
    # Independent loader generator: model initialization cannot change data order.
    train_loader = make_loader(cfg, "train", seed)
    val_loader = make_loader(cfg, "val", cfg["data_seed"])
    history, best_score = [], -math.inf
    best_records = best_summary = None
    best_epoch = None
    start = time.monotonic()
    for epoch in range(1, cfg["epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        diagnostics = {}
        train_loss = train_one_epoch(model, optimizer, train_loader, device=cfg["device"],
                                     max_grad_norm=cfg["max_grad_norm"], diagnostics=diagnostics,
                                     diagnostics_every=cfg.get("diagnostics_every", 10))
        records, val_loss = collect_candidates(model, val_loader, cfg["device"], cfg["min_score"])
        summary = summarize_cfg(records, cfg)
        score = objective_score(summary, cfg["objective"])
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        "val_loss": val_loss, "objective": score,
                        "best_f1": [m["f1"] for m in summary["best_by_channel"]],
                        "thresholds": [m["threshold"] for m in summary["best_by_channel"]],
                        "ap_at_floor": summary["ap_at_floor"],
                        "fusion_diagnostics": diagnostics})
        if scheduler is not None:
            scheduler.step()
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_records, best_summary = records, summary
            checkpoint = {
                "schema": SCHEMA, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "epoch": epoch, "val_loss": val_loss, "validation_objective": score,
                "config": cfg, "optimizer_name": optimizer_name, "params": params,
                "model_config": model.model_config(),
                "seed": seed, "thresholds": [m["threshold"] for m in summary["best_by_channel"]],
            }
            tmp = directory / "best.pt.tmp"
            torch.save(checkpoint, tmp)
            tmp.replace(directory / "best.pt")
            write_json(directory / "best_validation.json", summary)
        write_json(directory / "history.json", history)
        diagnostic_text = (f" gamma={diagnostics['gamma_end']:.5f} "
                           f"branch/radar={diagnostics['relative_branch_norm_mean']:.4g}"
                           if diagnostics.get("enabled") else "")
        print(f"{cfg['variant']}/{optimizer_name} seed={seed} epoch={epoch}/{cfg['epochs']} "
              f"train={train_loss:.5f} val={val_loss:.5f} {cfg['objective']}={score:.5f} "
              f"best={best_score:.5f}{diagnostic_text}", flush=True)
        if report is not None:
            report(epoch, best_score)
    save_cache(directory / "validation_candidates.json.gz", best_records, cfg, "val")
    thresholds = [m["threshold"] for m in best_summary["best_by_channel"]]
    selection = {"schema": SCHEMA, "checkpoint": "best.pt", "seed": seed,
                 "epoch": best_epoch, "validation_objective": best_score,
                 "thresholds": thresholds, "threshold_source": "validation",
                 "checkpoint_sha256": file_sha256(directory / "best.pt"),
                 "wall_seconds": time.monotonic()-start, "config": cfg,
                 "optimizer_name": optimizer_name, "params": params}
    write_json(directory / "selection.json", selection)
    from evaluation import plot_loss_curves, plot_pr_curves
    plot_loss_curves([h["train_loss"] for h in history], [h["val_loss"] for h in history],
                     str(directory / "loss_curves.png"),
                     title=f"{cfg['variant']} / {optimizer_name} / seed {seed} (line: min val loss)")
    for channel, name in enumerate(("vessel", "structure")):
        plot_pr_curves({cfg["variant"]: best_summary["sweep"]}, channel,
                       str(directory / f"validation_pr_{name}.png"),
                       title=f"Validation {name}: grid star; exact best F1 in JSON")
    del model, optimizer, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return selection


def validate_space(space):
    if set(space) != set(OPTIMIZERS):
        raise ValueError(f"Search space must contain exactly {OPTIMIZERS}")
    for name, entry in space.items():
        low, high = entry["lr"]
        if not 0 < low <= high or not entry["weight_decay"]:
            raise ValueError(f"Invalid space for {name}")
        if any(not math.isfinite(x) or x < 0 for x in entry["weight_decay"]):
            raise ValueError(f"Invalid weight decay for {name}")
        if name == "adam" and entry["weight_decay"] != [0.0]:
            raise ValueError("Adam is the zero-weight-decay control")


def build_cfg(args):
    labels = args.labels or str(Path(args.data_dir)/"labels.csv")
    cfg = {"data_dir": str(Path(args.data_dir).resolve()), "labels_path": str(Path(labels).resolve()),
           "scenes": {"train": config.TRAIN_SCENES, "val": config.VAL_SCENES, "test": config.TEST_SCENES},
           "data_seed": args.data_seed, "epochs": args.epochs, "batch_size": args.batch_size,
           "workers": args.workers, "device": args.device, "smoke": args.smoke,
           "scheduler": args.scheduler, "max_grad_norm": args.max_grad_norm,
           "cgcam_mode": getattr(args, "cgcam_mode", None) or "global",
           "gamma_init": args.gamma_init if getattr(args, "gamma_init", None) is not None else 0.0,
           "diagnostics_every": getattr(args, "diagnostics_every", 10),
           "objective": args.objective, "min_score": args.min_score,
           "threshold_min": args.threshold_min, "threshold_max": args.threshold_max,
           "match_distance": config.MATCH_DISTANCE_PIXELS,
           "max_train_samples": args.max_train_samples, "max_val_samples": args.max_val_samples,
           "code_sha256": code_digest(), "torch_version": str(torch.__version__),
           "optuna_version": str(optuna.__version__), "decay_policy": "weights_only_no_bias_norm_gamma"}
    for split, scenes in cfg["scenes"].items():
        other = [s for k, ss in cfg["scenes"].items() if k != split for s in ss]
        if set(scenes) & set(other):
            raise ValueError("Train, validation and test scenes must be disjoint")
    cfg["training_data_fingerprint"] = dataset_fingerprint(cfg, ("train", "val"))
    return cfg


def run_search(args):
    space = json.loads(Path(args.search_space).read_text()) if args.search_space else SEARCH_SPACES
    validate_space(space)
    cfg = build_cfg(args)
    storage = client_storage(args.storage, args.grpc_host, args.grpc_port)
    outputs = []
    for variant in args.variants:
        for name in args.optimizers:
            study_name = f"{args.study_prefix}__{variant}__{name}"
            directory = Path(args.output).resolve()/study_name
            contract = {"schema": SCHEMA, "config": dict(cfg, variant=variant),
                        "optimizer_name": name, "space": space[name], "seeds": args.seeds,
                        "sampler_seed": args.sampler_seed, "pruning": args.prune,
                        "artifact_root": str(directory)}
            with study_lock(directory):
                study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize",
                    load_if_exists=True,
                    sampler=optuna.samplers.TPESampler(seed=args.sampler_seed, n_startup_trials=5),
                    pruner=(optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
                            if args.prune else optuna.pruners.NopPruner()))
                previous = study.user_attrs.get("contract")
                if previous is not None and previous != contract:
                    raise ValueError(f"Study {study_name} has a different data/config/code contract. Use a NEW --study-prefix.")
                if previous is None and study.trials:
                    raise ValueError("Refusing to mix an unversioned preexisting study with this experiment")
                study.set_user_attr("contract", contract)
                write_json(directory/"contract.json", contract)
                # The OS lock proves no local worker with this artifact root is alive.
                # Recover abandoned trials after SIGKILL/reboot; epochs are not resumed.
                for old in study.trials:
                    if old.state == optuna.trial.TrialState.RUNNING:
                        study.tell(old.number, state=optuna.trial.TrialState.FAIL)
                        print(f"Marked abandoned trial {old.number} FAIL", flush=True)
                finished = [t for t in study.trials if t.state.is_finished()]
                study.sampler = optuna.samplers.TPESampler(seed=args.sampler_seed + len(finished), n_startup_trials=5)
                if not study.trials:
                    starters = [DEFAULTS[name]]
                    if name == "lion":
                        starters.append({"lr": 1e-5, "weight_decay": 0.1})
                    for starter in starters:
                        if (space[name]["lr"][0] <= starter["lr"] <= space[name]["lr"][1]
                                and starter["weight_decay"] in space[name]["weight_decay"]):
                            study.enqueue_trial(starter)

                def objective(trial):
                    low, high = space[name]["lr"]
                    params = {"lr": trial.suggest_float("lr", low, high, log=True),
                              "weight_decay": trial.suggest_categorical("weight_decay", space[name]["weight_decay"])}
                    trial_dir = directory/f"trial_{trial.number:05d}"
                    scores, runs = [], []
                    try:
                        for index, seed in enumerate(args.seeds):
                            def report(epoch, best):
                                trial.report((sum(scores)+best)/(len(scores)+1), index*args.epochs+epoch)
                                if trial.should_prune():
                                    raise optuna.TrialPruned(f"Pruned after seed {seed}, epoch {epoch}")
                            selection = fit_one(contract["config"], name, params, seed,
                                                trial_dir/f"seed_{seed}", report)
                            scores.append(selection["validation_objective"])
                            runs.append({"seed": seed, "selection": f"seed_{seed}/selection.json"})
                        result = {"trial": trial.number, "optimizer_name": name, "variant": variant,
                                  "params": params, "validation_objective": aggregate(scores), "runs": runs,
                                  "config": contract["config"]}
                        write_json(trial_dir/"result.json", result)
                        trial.set_user_attr("result_file", str(trial_dir/"result.json"))
                        trial.set_user_attr("seed_scores", scores)
                        trial.set_user_attr("std", result["validation_objective"]["std"])
                        return statistics.mean(scores)
                    except (FloatingPointError, torch.OutOfMemoryError) as exc:
                        trial.set_user_attr("failure_reason", str(exc))
                        raise
                    finally:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                remaining = max(0, args.trials-len(finished))
                try:
                    if remaining:
                        study.optimize(objective, n_trials=remaining, n_jobs=1,
                                       catch=(FloatingPointError, torch.OutOfMemoryError), gc_after_trial=True)
                finally:
                    rows = [{"number": t.number, "state": t.state.name, "value": t.value,
                             **t.params, "seed_scores": t.user_attrs.get("seed_scores"),
                             "failure_reason": t.user_attrs.get("failure_reason")} for t in study.trials]
                    write_json(directory/"trials.json", rows)
                    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    if complete:
                        best = study.best_trial
                        result = json.loads(Path(best.user_attrs["result_file"]).read_text())
                        # Portable relative links to the selected trial's per-seed artifacts.
                        manifest = dict(result, study_name=study_name,
                            runs=[{"seed": r["seed"], "selection": f"trial_{best.number:05d}/{r['selection']}"}
                                  for r in result["runs"]])
                        write_json(directory/"best.json", manifest)
                        outputs.append({"study": study_name, "best_value": best.value,
                                        "params": best.params, "std": best.user_attrs.get("std"),
                                        "manifest": str(directory/"best.json")})
                        print(f"BEST {study_name}: {best.value:.5f}; {directory/'best.json'}", flush=True)
    # This invocation's studies only; per-study files remain authoritative on resume.
    write_json(Path(args.output)/f"{args.study_prefix}_summary.json", outputs)


def run_repeat(args):
    source = Path(args.best).resolve()
    manifest = json.loads(source.read_text())
    if manifest.get("external_checkpoint"):
        raise ValueError("An external checkpoint has no selected optimizer configuration to repeat")
    cfg = dict(manifest["config"])
    if cfg["code_sha256"] != code_digest():
        raise ValueError("Code changed since search; use the matching code before repeating")
    if dataset_fingerprint(cfg, ("train", "val")) != cfg["training_data_fingerprint"]:
        raise ValueError("Training data changed since search")
    cfg["device"] = args.device
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Repeat output directory must be new/empty to preserve existing experiments")
    runs, scores = [], []
    for seed in args.seeds:
        selection = fit_one(cfg, manifest["optimizer_name"], manifest["params"], seed, out/f"seed_{seed}")
        runs.append({"seed": seed, "selection": f"seed_{seed}/selection.json"})
        scores.append(selection["validation_objective"])
    write_json(out/"best.json", {"schema": SCHEMA, "source_study": manifest.get("study_name"),
                                "optimizer_name": manifest["optimizer_name"], "variant": cfg["variant"],
                                "params": manifest["params"], "config": cfg, "runs": runs,
                                "validation_objective": aggregate(scores)})
    print(f"Repeated fixed configuration: {out/'best.json'}")


def run_calibrate(args):
    cache = load_cache(args.cache)
    if cache["split"] != "val":
        raise ValueError("Threshold selection is allowed on validation caches only")
    summary = summarize(cache["records"], min_score=cache["candidate_floor"],
                        threshold_min=args.threshold_min, threshold_max=args.threshold_max,
                        distance_threshold=args.match_distance)
    write_json(args.output, {"schema": SCHEMA, "threshold_source": "validation",
                            "cache_sha256": file_sha256(args.cache),
                            "thresholds": [m["threshold"] for m in summary["best_by_channel"]],
                            "summary": summary})
    print(f"Exact validation F1 thresholds: {[m['threshold'] for m in summary['best_by_channel']]}")
    print("This standalone report does not overwrite a checkpoint's frozen selection.json.")


def run_calibrate_checkpoint(args):
    """Calibrate an existing legacy checkpoint without retraining or loading test data."""
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Expected a project checkpoint with model_state_dict")
    saved = checkpoint.get("model_config") or {}
    prior_cfg = checkpoint.get("config", {})
    saved_variant = saved.get("fusion_mode", prior_cfg.get("variant"))
    if saved_variant is not None and saved_variant != args.variant:
        raise ValueError("--variant differs from the saved checkpoint's variant")
    saved_mode = saved.get("cgcam_mode", prior_cfg.get("cgcam_mode"))
    if args.cgcam_mode is not None and saved_mode is not None and args.cgcam_mode != saved_mode:
        raise ValueError("--cgcam-mode differs from the saved checkpoint's architecture")
    args.cgcam_mode = args.cgcam_mode or saved_mode or "global"
    if args.gamma_init is None:
        args.gamma_init = saved.get("gamma_init", prior_cfg.get("gamma_init", 0.0))
    cfg = dict(build_cfg(args), variant=args.variant)
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Calibration output must be new/empty")
    out.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    records, loss = collect_candidates(model, make_loader(cfg, "val", cfg["data_seed"]),
                                       cfg["device"], cfg["min_score"])
    summary = summarize_cfg(records, cfg)
    thresholds = [m["threshold"] for m in summary["best_by_channel"]]
    shutil.copyfile(args.checkpoint, out/"best.pt")
    save_cache(out/"validation_candidates.json.gz", records, cfg, "val")
    write_json(out/"best_validation.json", summary)
    selection = {"schema": SCHEMA, "checkpoint": "best.pt", "seed": args.seed,
                 "epoch": checkpoint.get("epoch"), "epoch_convention": "copied_from_external_checkpoint",
                 "validation_objective": objective_score(summary, cfg["objective"]),
                 "val_loss": loss, "thresholds": thresholds, "threshold_source": "validation",
                 "checkpoint_sha256": file_sha256(out/"best.pt"), "config": cfg,
                 "optimizer_name": "external_checkpoint", "params": {}}
    write_json(out/"selection.json", selection)
    write_json(out/"best.json", {"schema": SCHEMA, "external_checkpoint": True,
                                "config": cfg, "variant": args.variant,
                                "optimizer_name": "external_checkpoint", "params": {},
                                "runs": [{"seed": args.seed, "selection": "selection.json"}]})
    print(f"Calibrated existing checkpoint without retraining: {out/'best.json'}")


def run_evaluate(args):
    if not args.confirm_frozen:
        raise ValueError("Use --confirm-frozen only after optimizer, model and thresholds are selected on validation")
    source = Path(args.best).resolve()
    manifest = json.loads(source.read_text())
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Evaluation output must be new/empty; previous test reports will not be overwritten")
    out.mkdir(parents=True, exist_ok=True)
    all_results = []
    for run in manifest["runs"]:
        selection_path = source.parent/run["selection"]
        selection = json.loads(selection_path.read_text())
        cfg = dict(selection["config"])
        if selection["threshold_source"] != "validation":
            raise ValueError("Threshold source must be validation")
        if cfg["code_sha256"] != code_digest():
            raise ValueError("Code changed after selection; use the original experiment code")
        checkpoint_path = selection_path.parent/selection["checkpoint"]
        if file_sha256(checkpoint_path) != selection["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed after validation")
        cfg["device"] = args.device
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_model(cfg)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_fingerprint = dataset_fingerprint(cfg, ("test",))
        test_loader = make_loader(cfg, "test", cfg["data_seed"])
        records, loss = collect_candidates(model, test_loader, cfg["device"], cfg["min_score"])
        frozen = at_frozen_thresholds(records, selection["thresholds"], cfg["match_distance"])
        fixed = at_frozen_thresholds(records, [0.3, 0.3], cfg["match_distance"])
        curves = [DetectionCurve(records, c, cfg["match_distance"]) for c in range(2)]
        sweep_thresholds = sorted(set([cfg["min_score"], 0.3, *selection["thresholds"],
                                      *[i/100 for i in range(1, 100) if i/100 >= cfg["min_score"]]]))
        result = {"seed": run["seed"], "variant": cfg["variant"], "optimizer": selection["optimizer_name"],
                  "model_config": model.model_config(),
                  "epoch": selection["epoch"], "smoke": cfg["smoke"], "test_loss": loss,
                  "scope": "sampled_patch_instances_not_scene_level", "num_patches": len(records),
                  "threshold_source": "validation", "thresholds": selection["thresholds"],
                  "frozen_threshold_metrics": frozen, "fixed_0_3": fixed,
                  "candidate_floor": cfg["min_score"], "ap_at_floor": [c.ap for c in curves],
                  "ap_definition": "non_interpolated_grouped_score_detection_AP_above_candidate_floor",
                  "gt_counts": [c.total_gt for c in curves], "test_data_fingerprint": test_fingerprint,
                  "sweep": [{"threshold": t, "metrics": [c.at(t) for c in curves]} for t in sweep_thresholds]}
        # Deliberately no test argmax-F1: operating points remain frozen.
        seed_dir = out/f"seed_{run['seed']}"
        write_json(seed_dir/"test_metrics.json", result)
        save_cache(seed_dir/"test_candidates.json.gz", records, cfg, "test")
        plot_frozen_pr(result, seed_dir)
        all_results.append(result)
        del model, test_loader
    aggregate_results = []
    for channel, name in enumerate(("vessel", "structure")):
        row = {"channel": name}
        for metric in ("precision", "recall", "f1", "near_shore_fp"):
            row[metric] = aggregate([r["frozen_threshold_metrics"][channel][metric] for r in all_results])
        ap_values = [r["ap_at_floor"][channel] for r in all_results]
        row["ap_at_floor"] = aggregate(ap_values) if all(a is not None for a in ap_values) else None
        aggregate_results.append(row)
    write_json(out/"summary.json", {"source": str(source), "num_seeds": len(all_results),
                                    "threshold_source": "validation", "channels": aggregate_results,
                                    "scope": "sampled_patch_instances_not_scene_level"})
    with open(out/"metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "channel", "threshold", "precision", "recall", "f1", "tp", "fp", "fn", "near_shore_fp", "ap_at_floor"])
        writer.writeheader()
        for r in all_results:
            for c in range(2):
                m = r["frozen_threshold_metrics"][c]
                writer.writerow({"seed": r["seed"], "channel": c, "ap_at_floor": r["ap_at_floor"][c],
                                 **{key: m[key] for key in writer.fieldnames if key in m}})
    print(f"Frozen-threshold test report: {out/'summary.json'}")


def plot_frozen_pr(result, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for c, name in enumerate(("vessel", "structure")):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([p["metrics"][c]["recall"] for p in result["sweep"]],
                [p["metrics"][c]["precision"] for p in result["sweep"]])
        point = result["frozen_threshold_metrics"][c]
        ax.scatter(point["recall"], point["precision"], marker="*", s=130,
                   label=f"Validation-selected threshold {point['threshold']:.4f}")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1.02), ylim=(0, 1.02),
               title=f"Test {name}: {result['variant']} / {result['optimizer']}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(directory/f"test_pr_{name}.png", dpi=200)
        plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search", help="Train Optuna trials using train/validation data only")
    s.add_argument("--variants", nargs="+", choices=FUSION_MODES, default=list(FUSION_MODES))
    s.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    s.add_argument("--cgcam-mode", choices=CGCAM_MODES, default="global",
                   help="CGCAM fusion mechanism; ignored by SAR-only and early fusion")
    s.add_argument("--gamma-init", type=float, default=0.0,
                   help="Learnable residual scale initialization; compare 0 and 0.1")
    s.add_argument("--diagnostics-every", type=int, default=10,
                   help="Sample fusion activity/gradients every N train batches; 0 disables")
    s.add_argument("--study-prefix", default="darkvessel-v1")
    s.add_argument("--trials", type=int, default=10, help="Total finished trial budget PER variant/optimizer study")
    s.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    s.add_argument("--seeds", nargs="+", type=int, default=[42])
    s.add_argument("--sampler-seed", type=int, default=123)
    s.add_argument("--data-seed", type=int, default=42)
    s.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    s.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    s.add_argument("--device", default=str(config.DEVICE))
    s.add_argument("--cpu-threads", type=int, default=4)
    s.add_argument("--data-dir", default=config.DATA_DIR)
    s.add_argument("--labels")
    s.add_argument("--output", default="tuning_runs")
    s.add_argument("--storage", default="sqlite:///tuning_runs/optuna.db")
    s.add_argument("--grpc-host", help="Use the optuna-server storage proxy instead of direct SQL")
    s.add_argument("--grpc-port", type=int, default=13000)
    s.add_argument("--search-space", help="JSON overriding the documented optimizer search spaces")
    s.add_argument("--objective", choices=("vessel-f1", "macro-f1", "vessel-ap"), default="vessel-f1")
    s.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    s.add_argument("--max-grad-norm", type=float, default=1.0)
    s.add_argument("--min-score", type=float, default=0.01)
    s.add_argument("--threshold-min", type=float, default=0.01)
    s.add_argument("--threshold-max", type=float, default=0.95)
    s.add_argument("--prune", action="store_true", help="Opt-in median pruning; off for fair full-budget comparisons")
    s.add_argument("--max-train-samples", type=int, default=0)
    s.add_argument("--max-val-samples", type=int, default=0)
    s.add_argument("--smoke", action="store_true", help="Synthetic 32x32 data; NEVER use as scientific results")
    r = sub.add_parser("repeat", help="Retrain a selected fixed configuration with new seeds")
    r.add_argument("--best", required=True)
    r.add_argument("--seeds", nargs="+", type=int, required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--device", default=str(config.DEVICE))
    r.add_argument("--cpu-threads", type=int, default=4)
    e = sub.add_parser("evaluate", help="Test selected checkpoints at frozen validation thresholds")
    e.add_argument("--best", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--device", default=str(config.DEVICE))
    e.add_argument("--cpu-threads", type=int, default=4)
    e.add_argument("--confirm-frozen", action="store_true")
    c = sub.add_parser("calibrate", help="Repeat exact F1 threshold search on a validation cache, no training")
    c.add_argument("--cache", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--threshold-min", type=float, default=0.01)
    c.add_argument("--threshold-max", type=float, default=0.95)
    c.add_argument("--match-distance", type=float, default=config.MATCH_DISTANCE_PIXELS)
    x = sub.add_parser("calibrate-checkpoint", help="Select validation F1 cutoffs for an existing project checkpoint")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--variant", required=True, choices=FUSION_MODES)
    x.add_argument("--cgcam-mode", choices=CGCAM_MODES, default=None,
                   help="Normally inferred from checkpoint metadata; legacy default is global")
    x.add_argument("--gamma-init", type=float, default=None,
                   help="Initialization metadata only; saved learned gamma is restored")
    x.add_argument("--output", required=True)
    x.add_argument("--seed", type=int, default=42, help="Metadata: seed used to train the existing checkpoint")
    x.add_argument("--data-dir", default=config.DATA_DIR)
    x.add_argument("--labels")
    x.add_argument("--data-seed", type=int, default=42)
    x.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    x.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    x.add_argument("--device", default=str(config.DEVICE))
    x.add_argument("--cpu-threads", type=int, default=4)
    x.add_argument("--min-score", type=float, default=0.01)
    x.add_argument("--threshold-min", type=float, default=0.01)
    x.add_argument("--threshold-max", type=float, default=0.95)
    x.set_defaults(smoke=False, epochs=0, scheduler="none", max_grad_norm=1.0,
                   objective="vessel-f1", max_train_samples=0, max_val_samples=0)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if getattr(args, "gamma_init", None) is not None and not math.isfinite(args.gamma_init):
        p.error("--gamma-init must be finite")
    if getattr(args, "diagnostics_every", 0) < 0:
        p.error("--diagnostics-every must be non-negative")
    if hasattr(args, "cpu_threads"):
        if args.cpu_threads < 1:
            p.error("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
    if hasattr(args, "seeds") and (len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds)):
        p.error("Seeds must be distinct integers in [0, 2**32)")
    if args.command == "search":
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.study_prefix):
            p.error("Use letters, digits, underscores and hyphens in --study-prefix")
        if min(args.trials, args.epochs, args.batch_size) < 1 or args.workers < 0:
            p.error("Trials, epochs and batch size must be positive; workers non-negative")
        if min(args.max_train_samples, args.max_val_samples) < 0 or args.max_grad_norm <= 0:
            p.error("Sample limits must be non-negative and max-grad-norm positive")
        if not 0 < args.min_score <= min(args.threshold_min, 0.3) <= args.threshold_max < 1:
            p.error("Require 0 < min-score <= threshold-min <= threshold-max < 1, and min-score <= 0.3")
        if args.min_score > args.threshold_min or args.threshold_min > args.threshold_max:
            p.error("Invalid threshold range")
        if args.smoke:
            if not args.study_prefix.startswith("smoke"):
                p.error("Synthetic smoke runs require a --study-prefix starting with 'smoke'")
            args.workers = 0
            args.batch_size = min(args.batch_size, 2)
            print("SYNTHETIC SMOKE RUN: scores are not scientific results", flush=True)
    {"search": run_search, "repeat": run_repeat, "evaluate": run_evaluate,
     "calibrate": run_calibrate, "calibrate-checkpoint": run_calibrate_checkpoint}[args.command](args)


if __name__ == "__main__":
    main()
