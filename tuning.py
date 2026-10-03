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
import warnings

import optuna
import torch

import globals as config
from network import DarkVesselNet, FUSION_MODES
from optimizers import OPTIMIZERS, SEARCH_SPACES, WARM_STARTS, make_optimizer
from optuna_server import client_storage
from train import train_one_epoch
from tuning_metrics import DetectionCurve, summarize
from tuning_runtime import (code_digest, collect_candidates, dataset_fingerprint,
                            fd_limits, file_sha256, is_fd_exhaustion, iter_candidate_records,
                            make_loader, open_fd_count, raise_fd_limit, shutdown_loader)

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
        header = {"schema": SCHEMA, "split": split, "candidate_floor": cfg["min_score"],
                  "smoke": cfg["smoke"]}
        f.write(json.dumps(header, allow_nan=False)[:-1] + ', "records": [')
        # Convert one patch at a time, only when saving a selected epoch/test cache.
        for index, record in enumerate(iter_candidate_records(records)):
            if index:
                f.write(", ")
            json.dump(record, f, allow_nan=False)
        f.write("]}")
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


def fit_one(cfg, optimizer_name, params, seed, directory, report=None):
    """Train and choose epoch + per-class cutoffs entirely on validation data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config.set_seed(seed, deterministic=cfg.get("deterministic", False))
    model = DarkVesselNet(fusion_mode=cfg["variant"]).to(cfg["device"])
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
    try:
        for epoch in range(1, cfg["epochs"] + 1):
            lr = optimizer.param_groups[0]["lr"]
            epoch_start = time.monotonic()
            train_loss = train_one_epoch(model, optimizer, train_loader, device=cfg["device"],
                                         max_grad_norm=cfg["max_grad_norm"])
            # Include queued training kernels in the training timing on accelerators.
            if str(cfg["device"]).startswith("cuda"):
                torch.cuda.synchronize(cfg["device"])
            elif str(cfg["device"]).startswith("mps"):
                torch.mps.synchronize()
            train_seconds = time.monotonic() - epoch_start
            validation_start = time.monotonic()
            records, val_loss = collect_candidates(model, val_loader, cfg["device"], cfg["min_score"], compact=True)
            validation_seconds = time.monotonic() - validation_start
            metrics_start = time.monotonic()
            summary = summarize_cfg(records, cfg)
            score = objective_score(summary, cfg["objective"])
            metrics_seconds = time.monotonic() - metrics_start
            candidate_count = sum(len(p) for record in records for p in record["pred"])
            open_fds = open_fd_count()
            history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                            "val_loss": val_loss, "objective": score,
                            "train_seconds": train_seconds, "validation_seconds": validation_seconds,
                            "metrics_seconds": metrics_seconds, "candidate_count": candidate_count,
                            "validation_patches": len(records), "open_fds": open_fds,
                            "best_f1": [m["f1"] for m in summary["best_by_channel"]],
                            "thresholds": [m["threshold"] for m in summary["best_by_channel"]],
                            "ap_at_floor": summary["ap_at_floor"]})
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
                    "seed": seed, "thresholds": [m["threshold"] for m in summary["best_by_channel"]],
                }
                tmp = directory / "best.pt.tmp"
                torch.save(checkpoint, tmp)
                tmp.replace(directory / "best.pt")
                write_json(directory / "best_validation.json", summary)
            write_json(directory / "history.json", history)
            print(f"{cfg['variant']}/{optimizer_name} seed={seed} epoch={epoch}/{cfg['epochs']} "
                  f"train={train_loss:.5f} val={val_loss:.5f} {cfg['objective']}={score:.5f} "
                  f"best={best_score:.5f} | train={train_seconds:.1f}s "
                  f"validation={validation_seconds:.1f}s metrics={metrics_seconds:.2f}s "
                  f"candidates={candidate_count} fds={open_fds}", flush=True)
            if report is not None:
                report(epoch, best_score)
    finally:
        # Deterministic cleanup even when pruning/NaN/OOM/Ctrl-C raises: the
        # exception traceback would otherwise keep both loaders' persistent
        # workers (processes, pipes, memmaps) alive into the next trial.
        shutdown_loader(train_loader)
        shutdown_loader(val_loader)
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


def validate_space(space, optimizers=OPTIMIZERS):
    missing = [name for name in optimizers if name not in space]
    if missing:
        raise ValueError(f"Search space has no entry for {missing}")
    unknown = set(space) - set(OPTIMIZERS)
    if unknown:
        raise ValueError(f"Unknown optimizers in search space: {sorted(unknown)}")
    for name in optimizers:
        entry = space[name]
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
           "deterministic": getattr(args, "deterministic", False),
           "scheduler": args.scheduler, "max_grad_norm": args.max_grad_norm,
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


class FdExhausted(RuntimeError):
    """The process ran out of file descriptors; the trial is retried after a restart."""


MAX_AUTO_RESTARTS = 20
RESTART_ENV = "DVD_TUNING_RESTARTS"


def param_names(name):
    return f"{name}_lr", f"{name}_weight_decay"


def make_sampler(seed, startup_trials):
    # multivariate+group lets TPE model each optimizer's conditional (lr, decay)
    # subspace separately inside one study, instead of treating them independently.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=startup_trials,
                                          multivariate=True, group=True)


def balanced_plan(optimizers, space, min_trials, warm_start):
    """Round-robin queue guaranteeing every optimizer `min_trials` trials.

    Warm starts fix all parameters; the remaining entries only fix the optimizer
    so the sampler chooses (lr, decay) within that optimizer's domain.
    """
    plan, counts = [], dict.fromkeys(optimizers, 0)
    if warm_start:
        for name in optimizers:
            start = WARM_STARTS.get(name)
            if start is None:
                continue
            low, high = space[name]["lr"]
            if low <= start["lr"] <= high and start["weight_decay"] in space[name]["weight_decay"]:
                lr_key, wd_key = param_names(name)
                plan.append({"optimizer": name, lr_key: start["lr"], wd_key: start["weight_decay"]})
                counts[name] += 1
            else:
                print(f"Warm start for {name} lies outside the search space; skipped", flush=True)
    for _ in range(min_trials):
        for name in optimizers:
            if counts[name] < min_trials:
                plan.append({"optimizer": name})
                counts[name] += 1
    return plan


def trial_row(t):
    name = t.params.get("optimizer")
    lr_key, wd_key = param_names(name) if name else (None, None)
    return {"number": t.number, "state": t.state.name, "value": t.value, "optimizer": name,
            "lr": t.params.get(lr_key), "weight_decay": t.params.get(wd_key),
            "seed_scores": t.user_attrs.get("seed_scores"), "std": t.user_attrs.get("std"),
            "diverged_seeds": t.user_attrs.get("diverged_seeds"),
            "failure_reason": t.user_attrs.get("failure_reason"),
            "retry_of": t.user_attrs.get("retry_of"),
            "open_fds_start": t.user_attrs.get("open_fds_start"),
            "open_fds_end": t.user_attrs.get("open_fds_end")}


def usable(t):
    """COMPLETE trials whose every seed produced a selectable checkpoint."""
    return (t.state == optuna.trial.TrialState.COMPLETE and "result_file" in t.user_attrs
            and not t.user_attrs.get("diverged_seeds"))


def trial_manifest(t, study_name):
    result = json.loads(Path(t.user_attrs["result_file"]).read_text())
    # Portable relative links to the selected trial's per-seed artifacts.
    return dict(result, study_name=study_name,
                runs=[{"seed": r["seed"], "selection": f"trial_{t.number:05d}/{r['selection']}"}
                      for r in result["runs"]])


def write_study_outputs(study, directory, study_name, optimizers):
    trials = study.trials
    write_json(directory/"trials.json", [trial_row(t) for t in trials])
    candidates = [t for t in trials if usable(t)]
    overall = None
    if candidates:
        best = max(candidates, key=lambda t: t.value)
        write_json(directory/"best.json", trial_manifest(best, study_name))
        overall = {"study": study_name, "best_value": best.value, "optimizer": best.params["optimizer"],
                   "params": trial_manifest(best, study_name)["params"], "std": best.user_attrs.get("std"),
                   "manifest": str(directory/"best.json")}
        print(f"BEST {study_name}: {best.params['optimizer']} {best.value:.5f}; {directory/'best.json'}", flush=True)
    rows = []
    for name in optimizers:
        mine = [t for t in trials if t.params.get("optimizer") == name]
        complete = [t for t in mine if t.state == optuna.trial.TrialState.COMPLETE]
        values = sorted((t.value for t in complete), reverse=True)
        good = [t for t in complete if usable(t)]
        row = {"optimizer": name, "complete": len(complete),
               "failed": sum(t.state == optuna.trial.TrialState.FAIL for t in mine),
               "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in mine),
               "diverged": sum(bool(t.user_attrs.get("diverged_seeds")) for t in complete),
               "best_value": values[0] if values else None,
               "top3_mean": statistics.mean(values[:3]) if values else None,
               "median": statistics.median(values) if values else None,
               "worst": values[-1] if values else None,
               "best_trial": None, "best_lr": None, "best_weight_decay": None,
               "best_seed_scores": None, "best_seed_std": None, "manifest": None}
        if good:
            best = max(good, key=lambda t: t.value)
            manifest = trial_manifest(best, study_name)
            path = directory/f"best_{name}.json"
            write_json(path, manifest)
            row.update(best_trial=best.number, best_lr=manifest["params"]["lr"],
                       best_weight_decay=manifest["params"]["weight_decay"],
                       best_seed_scores=best.user_attrs.get("seed_scores"),
                       best_seed_std=best.user_attrs.get("std"), manifest=path.name)
        rows.append(row)
    ranked = sorted([r for r in rows if r["best_value"] is not None], key=lambda r: -r["best_value"])
    write_json(directory/"optimizer_comparison.json", {
        "study": study_name, "objective": "mean validation objective over the trial's seeds",
        "caveat": ("best_value is a maximum over a different number of noisy trials per optimizer; "
                   "confirm the ranking with `tuning.py compare` on fresh seeds before reporting it."),
        "ranking": [r["optimizer"] for r in ranked], "optimizers": rows})
    with open(directory/"optimizer_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    plot_optimizer_study(trials, optimizers, directory/"optimizer_comparison.png", study_name)
    return overall


def plot_optimizer_study(trials, optimizers, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, name in enumerate(optimizers):
        mine = [t for t in complete if t.params.get("optimizer") == name]
        if not mine:
            continue
        color = colors[i % len(colors)]
        values = [t.value for t in mine]
        jitter = [i + 0.12 * ((k % 5) - 2) / 2 for k in range(len(values))]
        left.scatter(jitter, values, color=color, alpha=0.75)
        left.hlines(max(values), i - 0.3, i + 0.3, color=color)
        right.scatter([t.params[param_names(name)[0]] for t in mine], values,
                      color=color, alpha=0.75, label=name)
    left.set_xticks(range(len(optimizers)), optimizers)
    left.set(ylabel="Validation objective (mean over seeds)", title="Trials per optimizer (line = best)")
    right.set(xscale="log", xlabel="Learning rate", title="Learning rate vs objective")
    right.legend()
    for ax in (left, right):
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def restart_process(reason):
    """Replace this process with a fresh copy of the same command.

    All study state lives in Optuna storage and on disk, so the new process
    resumes exactly where this one stopped, with a clean descriptor table.
    """
    count = int(os.environ.get(RESTART_ENV, "0")) + 1
    if count > MAX_AUTO_RESTARTS:
        raise RuntimeError(f"{reason}; already restarted {count-1} times. Investigate `fds=` in the "
                           "logs and the open_fds fields in history.json/trials.json.")
    os.environ[RESTART_ENV] = str(count)
    print(f"AUTO-RESTART {count}/{MAX_AUTO_RESTARTS}: {reason}. Re-executing the same command; "
          "the study resumes from storage.", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable] + sys.argv)


_FD_BASELINE = None


def fd_pressure(fraction):
    """Reason string when descriptors *grew* past `fraction` of the remaining headroom.

    Measured relative to the count at process start, so a restart (which resets
    the count to that baseline) can never loop on a merely low limit.
    """
    global _FD_BASELINE
    soft, _ = fd_limits()
    count = open_fd_count()
    if not soft or count is None or soft <= 0:
        return None
    if _FD_BASELINE is None:
        _FD_BASELINE = count
    headroom = soft - _FD_BASELINE
    if headroom > 0 and count - _FD_BASELINE >= fraction * headroom:
        return (f"open file descriptors grew from {_FD_BASELINE} to {count} "
                f"(limit {soft}); restarting before the next trial")
    return None


def run_search(args):
    space = json.loads(Path(args.search_space).read_text()) if args.search_space else SEARCH_SPACES
    optimizers = list(dict.fromkeys(args.optimizers))
    validate_space(space, optimizers)
    if args.trials < args.min_trials_per_optimizer * len(optimizers):
        raise ValueError(f"--trials {args.trials} cannot give {len(optimizers)} optimizers "
                         f"{args.min_trials_per_optimizer} trials each; raise --trials or lower "
                         "--min-trials-per-optimizer")
    cfg = build_cfg(args)
    storage = client_storage(args.storage, args.grpc_host, args.grpc_port)
    outputs, restart_reason = [], None
    for variant in args.variants:
        study_name = f"{args.study_prefix}__{variant}"
        directory = Path(args.output).resolve()/study_name
        contract = {"schema": SCHEMA, "study_kind": "optimizer_comparison",
                    "config": dict(cfg, variant=variant), "optimizers": optimizers,
                    "space": {name: space[name] for name in optimizers}, "seeds": args.seeds,
                    "sampler_seed": args.sampler_seed, "startup_trials": args.startup_trials,
                    "pruning": args.prune, "min_trials_per_optimizer": args.min_trials_per_optimizer,
                    "warm_start": WARM_STARTS if args.warm_start else None,
                    "artifact_root": str(directory)}
        balanced = args.min_trials_per_optimizer * len(optimizers)
        with study_lock(directory):
            study = optuna.create_study(
                study_name=study_name, storage=storage, direction="maximize", load_if_exists=True,
                sampler=make_sampler(args.sampler_seed, args.startup_trials),
                # Optional and coarse: only after the balanced phase and only after a whole
                # first seed, because optimizers learn at very different speeds (v1 data).
                pruner=(optuna.pruners.MedianPruner(n_startup_trials=balanced, n_warmup_steps=args.epochs)
                        if args.prune else optuna.pruners.NopPruner()))
            previous = study.user_attrs.get("contract")
            if previous is not None and previous != contract:
                raise ValueError(f"Study {study_name} has a different data/config/code contract. Use a NEW --study-prefix.")
            if previous is None and study.trials:
                raise ValueError("Refusing to mix an unversioned preexisting study with this experiment")
            study.set_user_attr("contract", contract)
            write_json(directory/"contract.json", contract)
            # The OS lock proves no local worker with this artifact root is alive.
            # Recover abandoned trials after SIGKILL/reboot and queue one retry each.
            for old in study.trials:
                if old.state == optuna.trial.TrialState.RUNNING:
                    study.tell(old.number, state=optuna.trial.TrialState.FAIL)
                    if old.params.get("optimizer") and "retry_of" not in old.user_attrs:
                        study.enqueue_trial(old.params, user_attrs={"retry_of": old.number})
                    print(f"Marked abandoned trial {old.number} FAIL and queued a retry", flush=True)
            if not study.trials:
                for params in balanced_plan(optimizers, space, args.min_trials_per_optimizer, args.warm_start):
                    study.enqueue_trial(params)
            counted = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
            finished = [t for t in study.trials if t.state.is_finished()]
            study.sampler = make_sampler(args.sampler_seed + len(finished), args.startup_trials)

            ran = []

            def objective(trial):
                ran.append(trial.number)
                name = trial.suggest_categorical("optimizer", optimizers)
                lr_key, wd_key = param_names(name)
                low, high = space[name]["lr"]
                params = {"lr": trial.suggest_float(lr_key, low, high, log=True),
                          "weight_decay": trial.suggest_categorical(wd_key, space[name]["weight_decay"])}
                trial.set_user_attr("open_fds_start", open_fd_count())
                trial_dir = directory/f"trial_{trial.number:05d}"
                scores, runs, diverged = [], [], []
                try:
                    for index, seed in enumerate(args.seeds):
                        def report(epoch, best):
                            trial.report((sum(scores)+best)/(len(scores)+1), index*args.epochs+epoch)
                            if trial.should_prune():
                                raise optuna.TrialPruned(f"Pruned after seed {seed}, epoch {epoch}")
                        try:
                            selection = fit_one(contract["config"], name, params, seed,
                                                trial_dir/f"seed_{seed}", report)
                        except FloatingPointError as exc:
                            # Divergence is a property of the configuration: score it as 0 so
                            # TPE learns to avoid it, rather than a FAIL that TPE ignores.
                            diverged.append({"seed": seed, "reason": str(exc)})
                            scores.append(0.0)
                            print(f"Trial {trial.number} seed {seed} diverged: {exc}", flush=True)
                            continue
                        scores.append(selection["validation_objective"])
                        runs.append({"seed": seed, "selection": f"seed_{seed}/selection.json"})
                    result = {"trial": trial.number, "optimizer_name": name, "variant": variant,
                              "params": params, "validation_objective": aggregate(scores), "runs": runs,
                              "diverged_seeds": diverged, "config": contract["config"]}
                    write_json(trial_dir/"result.json", result)
                    trial.set_user_attr("result_file", str(trial_dir/"result.json"))
                    trial.set_user_attr("seed_scores", scores)
                    trial.set_user_attr("std", result["validation_objective"]["std"])
                    if diverged:
                        trial.set_user_attr("diverged_seeds", [d["seed"] for d in diverged])
                    return statistics.mean(scores)
                except torch.OutOfMemoryError as exc:
                    trial.set_user_attr("failure_reason", f"out_of_memory: {exc}")
                    raise
                except optuna.TrialPruned:
                    raise
                except Exception as exc:
                    if is_fd_exhaustion(exc):
                        trial.set_user_attr("failure_reason", f"fd_exhaustion: {exc}")
                        raise FdExhausted(str(exc)) from exc
                    raise
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    trial.set_user_attr("open_fds_end", open_fd_count())

            failures = 0
            try:
                while True:
                    done = sum(t.state in counted for t in study.get_trials(deepcopy=False))
                    if done >= args.trials:
                        break
                    if failures >= args.max_failures:
                        print(f"Stopping {study_name}: {failures} failed trials in this run", flush=True)
                        break
                    if args.auto_restart and (reason := fd_pressure(args.fd_restart_fraction)):
                        restart_reason = reason
                        break
                    study.optimize(objective, n_trials=1, n_jobs=1, gc_after_trial=True,
                                   catch=(torch.OutOfMemoryError, FdExhausted))
                    # Not study.trials[-1]: queued retries can have higher numbers.
                    last = next(t for t in study.get_trials(deepcopy=False) if t.number == ran[-1])
                    if last.state == optuna.trial.TrialState.FAIL:
                        failures += 1
                        reason = last.user_attrs.get("failure_reason", "")
                        if reason.startswith("fd_exhaustion"):
                            if "retry_of" not in last.user_attrs:
                                study.enqueue_trial(last.params, user_attrs={"retry_of": last.number})
                            restart_reason = f"trial {last.number} hit 'too many open files'"
                            if args.auto_restart:
                                break
                    done_now = sum(t.state in counted for t in study.get_trials(deepcopy=False))
                    print(f"[{study_name}] {done_now}/{args.trials} trials done "
                          f"(balanced phase: {balanced}); open fds={open_fd_count()}", flush=True)
            finally:
                overall = write_study_outputs(study, directory, study_name, optimizers)
                if overall:
                    outputs.append(overall)
        if restart_reason and args.auto_restart:
            break
    # This invocation's studies only; per-study files remain authoritative on resume.
    write_json(Path(args.output)/f"{args.study_prefix}_summary.json", outputs)
    if restart_reason and args.auto_restart:
        restart_process(restart_reason)


def repeat_manifest(manifest, seeds, out, device):
    if manifest.get("external_checkpoint"):
        raise ValueError("An external checkpoint has no selected optimizer configuration to repeat")
    cfg = dict(manifest["config"])
    if cfg["code_sha256"] != code_digest():
        raise ValueError("Code changed since search; use the matching code before repeating")
    if dataset_fingerprint(cfg, ("train", "val")) != cfg["training_data_fingerprint"]:
        raise ValueError("Training data changed since search")
    cfg["device"] = device
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Repeat output directory {out} must be new/empty to preserve existing experiments")
    runs, scores = [], []
    for seed in seeds:
        selection = fit_one(cfg, manifest["optimizer_name"], manifest["params"], seed, out/f"seed_{seed}")
        runs.append({"seed": seed, "selection": f"seed_{seed}/selection.json"})
        scores.append(selection["validation_objective"])
    best = {"schema": SCHEMA, "source_study": manifest.get("study_name"),
            "source_trial": manifest.get("trial"),
            "optimizer_name": manifest["optimizer_name"], "variant": cfg["variant"],
            "params": manifest["params"], "config": cfg, "runs": runs,
            "seed_scores": scores, "validation_objective": aggregate(scores)}
    write_json(out/"best.json", best)
    return best


def run_repeat(args):
    manifest = json.loads(Path(args.best).resolve().read_text())
    repeat_manifest(manifest, args.seeds, args.output, args.device)
    print(f"Repeated fixed configuration: {Path(args.output).resolve()/'best.json'}")


def run_compare(args):
    """Confirm the study's per-optimizer winners on fresh, shared seeds.

    The search ranks optimizers by a maximum over unequal numbers of noisy
    trials (winner's curse). Retraining every optimizer's selected configuration
    on the same new seeds gives an unbiased, paired comparison.
    """
    study_dir = Path(args.study_dir).resolve()
    manifests = {}
    for path in sorted(study_dir.glob("best_*.json")):
        name = path.stem[len("best_"):]
        if name in OPTIMIZERS and (not args.optimizers or name in args.optimizers):
            manifests[name] = json.loads(path.read_text())
    if len(manifests) < 1:
        raise ValueError(f"No best_<optimizer>.json manifests in {study_dir}; run `search` first")
    reused = sorted(set(args.seeds) & {r["seed"] for m in manifests.values() for r in m["runs"]})
    if reused:
        print(f"WARNING: seeds {reused} were used during the search; fresh seeds give an unbiased check", flush=True)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, manifest in manifests.items():
        target = out/name
        if (target/"best.json").exists():
            done = json.loads((target/"best.json").read_text())
            if [r["seed"] for r in done["runs"]] != list(args.seeds) or done["params"] != manifest["params"]:
                raise ValueError(f"{target} holds a different comparison; use a new --output")
            print(f"Reusing finished comparison runs for {name}", flush=True)
            results[name] = done
            continue
        if target.exists() and any(target.iterdir()):
            # An interrupted earlier attempt: keep it for inspection, start clean.
            target.rename(out/f"{name}.interrupted.{int(time.time())}")
        print(f"Comparing {name}: lr={manifest['params']['lr']:.3g} "
              f"wd={manifest['params']['weight_decay']} seeds={args.seeds}", flush=True)
        results[name] = repeat_manifest(manifest, args.seeds, target, args.device)
    leader = max(results, key=lambda n: results[n]["validation_objective"]["mean"])
    rows = []
    for name, res in results.items():
        diffs = [a - b for a, b in zip(res["seed_scores"], results[leader]["seed_scores"])]
        rows.append({"optimizer": name, "lr": res["params"]["lr"],
                     "weight_decay": res["params"]["weight_decay"],
                     "search_value": manifests[name]["validation_objective"]["mean"],
                     "seeds": args.seeds, "seed_scores": res["seed_scores"],
                     "mean": res["validation_objective"]["mean"], "std": res["validation_objective"]["std"],
                     "paired_diff_vs_leader_mean": statistics.mean(diffs),
                     "paired_diff_vs_leader_std": statistics.stdev(diffs) if len(diffs) > 1 else None,
                     "manifest": f"{name}/best.json"})
    rows.sort(key=lambda r: -r["mean"])
    write_json(out/"comparison.json", {"study_dir": str(study_dir), "leader": leader,
                                      "seeds": args.seeds, "objective": "validation objective per fresh seed",
                                      "optimizers": rows})
    with open(out/"comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    plot_comparison(rows, out/"comparison.png", f"{study_dir.name}: fresh-seed confirmation")
    print(f"Optimizer comparison: {out/'comparison.json'}")
    for row in rows:
        std = f"{row['std']:.4f}" if row["std"] is not None else "n/a"
        print(f"  {row['optimizer']:8s} mean={row['mean']:.4f} std={std} "
              f"vs {leader}: {row['paired_diff_vs_leader_mean']:+.4f}", flush=True)


def plot_comparison(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, row in enumerate(rows):
        ax.scatter([i] * len(row["seed_scores"]), row["seed_scores"], alpha=0.7)
        ax.errorbar(i + 0.15, row["mean"], yerr=row["std"] or 0, fmt="o", color="black", capsize=4)
    ax.set_xticks(range(len(rows)), [r["optimizer"] for r in rows])
    ax.set(ylabel="Validation objective", title=title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    cfg = dict(build_cfg(args), variant=args.variant)
    config.set_seed(args.seed, deterministic=cfg["deterministic"])
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Calibration output must be new/empty")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Expected a project checkpoint with model_state_dict")
    model = DarkVesselNet(fusion_mode=args.variant).to(cfg["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loader = make_loader(cfg, "val", cfg["data_seed"])
    try:
        records, loss = collect_candidates(model, val_loader, cfg["device"], cfg["min_score"], compact=True)
    finally:
        shutdown_loader(val_loader)
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
        config.set_seed(selection["seed"], deterministic=cfg.get("deterministic", False))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = DarkVesselNet(fusion_mode=cfg["variant"]).to(cfg["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
        test_fingerprint = dataset_fingerprint(cfg, ("test",))
        test_loader = make_loader(cfg, "test", cfg["data_seed"])
        try:
            records, loss = collect_candidates(model, test_loader, cfg["device"], cfg["min_score"], compact=True)
        finally:
            shutdown_loader(test_loader)
        curves = [DetectionCurve(records, c, cfg["match_distance"]) for c in range(2)]
        frozen = [curve.at(selection["thresholds"][c]) for c, curve in enumerate(curves)]
        fixed = [curve.at(0.3) for curve in curves]
        sweep_thresholds = sorted(set([cfg["min_score"], 0.3, *selection["thresholds"],
                                      *[i/100 for i in range(1, 100) if i/100 >= cfg["min_score"]]]))
        result = {"seed": run["seed"], "variant": cfg["variant"], "optimizer": selection["optimizer_name"],
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
    s = sub.add_parser("search", help="One Optuna study per variant comparing optimizers (train/validation only)")
    s.add_argument("--variants", nargs="+", choices=FUSION_MODES, default=list(FUSION_MODES))
    s.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS),
                   help="Optimizers competing inside each variant's single study")
    s.add_argument("--study-prefix", default="darkvessel-v2")
    s.add_argument("--trials", type=int, default=30,
                   help="Total COMPLETE/PRUNED trial budget PER variant study, shared by all optimizers "
                        "(failed trials do not consume it)")
    s.add_argument("--min-trials-per-optimizer", type=int, default=4,
                   help="Balanced phase: every optimizer gets this many trials (round-robin) before "
                        "TPE may concentrate the remaining budget")
    warm = s.add_mutually_exclusive_group()
    warm.add_argument("--warm-start", dest="warm_start", action="store_true", default=True,
                      help="Start each optimizer at its full-v1 incumbent (default)")
    warm.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    s.add_argument("--startup-trials", type=int, default=10, help="Random TPE startup trials")
    s.add_argument("--max-failures", type=int, default=10,
                   help="Stop a study after this many FAIL trials in one invocation")
    s.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    s.add_argument("--seeds", nargs="+", type=int, default=[42, 43],
                   help="Training seeds per trial; the objective is their mean. One seed cannot "
                        "resolve differences below ~0.02 F1 (replicate noise in full-v1)")
    s.add_argument("--sampler-seed", type=int, default=123)
    s.add_argument("--data-seed", type=int, default=42)
    s.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    s.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    s.add_argument("--device", default=str(config.DEVICE))
    cudnn = s.add_mutually_exclusive_group()
    cudnn.add_argument("--fast-cudnn", dest="deterministic", action="store_false", default=False,
                       help="Enable cuDNN benchmarking and allow nondeterministic kernels (default)")
    cudnn.add_argument("--deterministic", action="store_true",
                       help="Opt into deterministic cuDNN algorithms and disable benchmarking")
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
    s.add_argument("--prune", action="store_true",
                   help="Opt-in median pruning after the balanced phase and the first full seed. Not "
                        "recommended: v1 runs at 0.3-0.6 F1 by epoch 10-15 still finished at 0.89-0.93")
    restart = s.add_mutually_exclusive_group()
    restart.add_argument("--auto-restart", dest="auto_restart", action="store_true", default=True,
                         help="On fd exhaustion or fd pressure, re-exec this command and resume (default)")
    restart.add_argument("--no-auto-restart", dest="auto_restart", action="store_false")
    s.add_argument("--fd-restart-fraction", type=float, default=0.5,
                   help="Restart before a trial once open fds have grown by this fraction of the headroom "
                        "between the process-start count and the soft limit")
    s.add_argument("--max-train-samples", type=int, default=0)
    s.add_argument("--max-val-samples", type=int, default=0)
    s.add_argument("--smoke", action="store_true", help="Synthetic 32x32 data; NEVER use as scientific results")
    r = sub.add_parser("repeat", help="Retrain a selected fixed configuration with new seeds")
    r.add_argument("--best", required=True)
    r.add_argument("--seeds", nargs="+", type=int, required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--device", default=str(config.DEVICE))
    r.add_argument("--cpu-threads", type=int, default=4)
    m = sub.add_parser("compare", help="Retrain each optimizer's selected config on fresh shared seeds and compare")
    m.add_argument("--study-dir", required=True, help="A search study directory containing best_<optimizer>.json")
    m.add_argument("--seeds", nargs="+", type=int, required=True)
    m.add_argument("--output", required=True)
    m.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, help="Subset (default: all in the study)")
    m.add_argument("--device", default=str(config.DEVICE))
    m.add_argument("--cpu-threads", type=int, default=4)
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
    x.add_argument("--output", required=True)
    x.add_argument("--seed", type=int, default=42, help="Metadata: seed used to train the existing checkpoint")
    x.add_argument("--data-dir", default=config.DATA_DIR)
    x.add_argument("--labels")
    x.add_argument("--data-seed", type=int, default=42)
    x.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    x.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    x.add_argument("--device", default=str(config.DEVICE))
    cudnn = x.add_mutually_exclusive_group()
    cudnn.add_argument("--fast-cudnn", dest="deterministic", action="store_false", default=False,
                       help="Enable cuDNN benchmarking and allow nondeterministic kernels (default)")
    cudnn.add_argument("--deterministic", action="store_true",
                       help="Opt into deterministic cuDNN algorithms and disable benchmarking")
    x.add_argument("--cpu-threads", type=int, default=4)
    x.add_argument("--min-score", type=float, default=0.01)
    x.add_argument("--threshold-min", type=float, default=0.01)
    x.add_argument("--threshold-max", type=float, default=0.95)
    x.set_defaults(smoke=False, epochs=0, scheduler="none", max_grad_norm=1.0,
                   objective="vessel-f1", max_train_samples=0, max_val_samples=0)
    for command in (s, r, m, e, x):
        command.add_argument("--sharing-strategy", choices=("file_system", "file_descriptor"),
                             default="file_system",
                             help="torch.multiprocessing tensor sharing between loader workers and "
                                  "this process. file_system avoids one open fd per in-flight tensor")
    return p


def setup_process(args):
    """Process-wide guards against 'OSError: [Errno 24] Too many open files'."""
    before, after = raise_fd_limit()
    strategy = getattr(args, "sharing_strategy", None)
    if strategy:
        import torch.multiprocessing as tmp
        if strategy in tmp.get_all_sharing_strategies():
            tmp.set_sharing_strategy(strategy)
    fd_pressure(1.0)  # record the post-setup descriptor baseline
    if before is not None:
        note = f"raised from {before}" if after != before else "unchanged"
        print(f"Open-file limit: {after} ({note}); sharing strategy: {strategy or 'default'}; "
              f"open fds now: {open_fd_count()}", flush=True)


def main():
    p = parser()
    args = p.parse_args()
    if hasattr(args, "cpu_threads"):
        if args.cpu_threads < 1:
            p.error("--cpu-threads must be positive")
        torch.set_num_threads(args.cpu_threads)
    if hasattr(args, "seeds") and (len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds)):
        p.error("Seeds must be distinct integers in [0, 2**32)")
    setup_process(args)
    if args.command == "search":
        if args.min_trials_per_optimizer < 0 or args.startup_trials < 0 or args.max_failures < 1:
            p.error("--min-trials-per-optimizer/--startup-trials must be >= 0 and --max-failures >= 1")
        if not 0 < args.fd_restart_fraction <= 1:
            p.error("--fd-restart-fraction must be in (0, 1]")
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
    {"search": run_search, "repeat": run_repeat, "compare": run_compare, "evaluate": run_evaluate,
     "calibrate": run_calibrate, "calibrate-checkpoint": run_calibrate_checkpoint}[args.command](args)


if __name__ == "__main__":
    main()
