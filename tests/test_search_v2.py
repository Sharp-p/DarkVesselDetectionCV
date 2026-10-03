"""Single-study optimizer comparison, refined spaces and file-descriptor hygiene."""
import errno
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import optuna
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tuning
from optimizers import OPTIMIZERS, SEARCH_SPACES, WARM_STARTS
from tuning_runtime import is_fd_exhaustion, open_fd_count

optuna.logging.set_verbosity(optuna.logging.WARNING)


def smoke_args(tmp_path, *extra, trials=6, optimizers=("adamw", "sgd"), seeds=("42",)):
    args = tuning.parser().parse_args([
        "search", "--smoke", "--study-prefix", "smoke-test", "--variants", "sar_only",
        "--optimizers", *optimizers, "--trials", str(trials), "--min-trials-per-optimizer", "2",
        "--epochs", "1", "--seeds", *seeds, "--device", "cpu", "--output", str(tmp_path/"runs"),
        "--storage", f"sqlite:///{tmp_path}/optuna.db", "--startup-trials", "2", *extra])
    args.workers, args.batch_size = 0, 2
    return args


def load_study(tmp_path):
    return optuna.load_study(study_name="smoke-test__sar_only", storage=f"sqlite:///{tmp_path}/optuna.db")


def fake_fit(score=0.5, fail=None):
    """fit_one stand-in; `fail(seed, call)` may raise to simulate a fault."""
    calls = []

    def fit(cfg, name, params, seed, directory, report=None):
        calls.append((name, dict(params), seed))
        if fail is not None:
            fail(seed, len(calls))
        Path(directory).mkdir(parents=True, exist_ok=True)
        if report is not None:
            report(1, score)
        return {"validation_objective": score}
    fit.calls = calls
    return fit


def test_refined_space_contains_every_warm_start():
    tuning.validate_space(SEARCH_SPACES)
    for name in OPTIMIZERS:
        low, high = SEARCH_SPACES[name]["lr"]
        assert low <= WARM_STARTS[name]["lr"] <= high
        assert WARM_STARTS[name]["weight_decay"] in SEARCH_SPACES[name]["weight_decay"]


def test_balanced_plan_is_round_robin_with_warm_starts_first():
    plan = tuning.balanced_plan(["adamw", "lion", "sgd"], SEARCH_SPACES, 3, warm_start=True)
    assert [p["optimizer"] for p in plan[:3]] == ["adamw", "lion", "sgd"]
    assert plan[0] == {"optimizer": "adamw", "adamw_lr": 1.5e-4, "adamw_weight_decay": 1e-2}
    assert all(set(p) == {"optimizer"} for p in plan[3:])
    assert [p["optimizer"] for p in plan[3:]] == ["adamw", "lion", "sgd"] * 2
    assert len(tuning.balanced_plan(["adam"], SEARCH_SPACES, 2, warm_start=False)) == 2


def test_one_study_trains_every_optimizer_and_writes_comparison(tmp_path):
    args = smoke_args(tmp_path, trials=5, optimizers=("adamw", "sgd", "lion"))
    args.min_trials_per_optimizer = 1
    tuning.run_search(args)
    study = load_study(tmp_path)
    assert {t.params["optimizer"] for t in study.trials} == {"adamw", "sgd", "lion"}
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    out = tmp_path/"runs"/"smoke-test__sar_only"
    comparison = json.loads((out/"optimizer_comparison.json").read_text())
    assert {r["optimizer"] for r in comparison["optimizers"]} == {"adamw", "sgd", "lion"}
    for name in ("adamw", "sgd", "lion"):
        manifest = json.loads((out/f"best_{name}.json").read_text())
        assert manifest["optimizer_name"] == name
        assert (out/manifest["runs"][0]["selection"]).exists()
    assert (out/"optimizer_comparison.png").exists()
    # Conditional parameter names keep each optimizer's lr domain separate.
    for t in study.trials:
        name = t.params["optimizer"]
        low, high = SEARCH_SPACES[name]["lr"]
        assert low <= t.params[f"{name}_lr"] <= high


def test_budget_too_small_for_balanced_phase_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="min-trials-per-optimizer"):
        tuning.run_search(smoke_args(tmp_path, trials=3))


def test_fd_exhaustion_fails_trial_and_retries_same_params(tmp_path, monkeypatch):
    def fail(seed, call):
        if call == 1:
            raise OSError(errno.EMFILE, "Too many open files")
    fit = fake_fit(fail=fail)
    monkeypatch.setattr(tuning, "fit_one", fit)
    tuning.run_search(smoke_args(tmp_path, "--no-auto-restart", trials=4))
    trials = load_study(tmp_path).trials
    assert trials[0].state == optuna.trial.TrialState.FAIL
    assert trials[0].user_attrs["failure_reason"].startswith("fd_exhaustion")
    retry = next(t for t in trials if t.user_attrs.get("retry_of") == 0)
    assert retry.state == optuna.trial.TrialState.COMPLETE and retry.params == trials[0].params
    assert sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials) == 4


def test_fd_exhaustion_auto_restarts_process(tmp_path, monkeypatch):
    def fail(seed, call):
        raise RuntimeError("Too many open files. Communication with the workers is no longer possible.")
    monkeypatch.setattr(tuning, "fit_one", fake_fit(fail=fail))
    calls = []

    class Exec(Exception):
        pass

    def fake_exec(path, argv):
        calls.append(argv)
        raise Exec()
    monkeypatch.setattr(tuning.os, "execv", fake_exec)
    monkeypatch.delenv(tuning.RESTART_ENV, raising=False)
    with pytest.raises(Exec):
        tuning.run_search(smoke_args(tmp_path, trials=4))
    assert calls and calls[0][0] == sys.executable
    assert os.environ[tuning.RESTART_ENV] == "1"
    monkeypatch.delenv(tuning.RESTART_ENV)
    # Outputs were written before re-exec, and the failed trial is queued again.
    assert (tmp_path/"runs"/"smoke-test__sar_only"/"trials.json").exists()
    states = [t.state for t in load_study(tmp_path).trials]
    assert optuna.trial.TrialState.WAITING in states


def test_divergence_scores_zero_and_cannot_be_selected(tmp_path, monkeypatch):
    def fail(seed, call):
        if seed == 43:
            raise FloatingPointError("Non-finite training loss")
    monkeypatch.setattr(tuning, "fit_one", fake_fit(score=0.8, fail=fail))
    tuning.run_search(smoke_args(tmp_path, "--no-auto-restart", trials=4, seeds=("42", "43")))
    trials = load_study(tmp_path).trials
    assert all(t.value == pytest.approx(0.4) for t in trials)
    assert all(t.user_attrs["diverged_seeds"] == [43] for t in trials)
    assert not (tmp_path/"runs"/"smoke-test__sar_only"/"best.json").exists()


def test_abandoned_running_trial_is_failed_and_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "fit_one", fake_fit())
    tuning.run_search(smoke_args(tmp_path, trials=4))
    study = load_study(tmp_path)
    crashed = study.ask({"optimizer": optuna.distributions.CategoricalDistribution(["adamw", "sgd"])})
    params = {"optimizer": crashed.params["optimizer"]}
    tuning.run_search(smoke_args(tmp_path, trials=5))
    trials = load_study(tmp_path).trials
    assert trials[crashed.number].state == optuna.trial.TrialState.FAIL
    retry = next(t for t in trials if t.user_attrs.get("retry_of") == crashed.number)
    assert retry.params["optimizer"] == params["optimizer"]
    assert retry.state == optuna.trial.TrialState.COMPLETE


def test_changed_search_space_needs_new_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "fit_one", fake_fit())
    tuning.run_search(smoke_args(tmp_path, trials=4))
    with pytest.raises(ValueError, match="NEW --study-prefix"):
        tuning.run_search(smoke_args(tmp_path, "--search-space", str(Path(tuning.__file__).parent/"tuning_search_space_v1.json"), trials=4))


def test_is_fd_exhaustion_walks_exception_chain():
    try:
        try:
            raise OSError(errno.EMFILE, "x")
        except OSError as inner:
            raise RuntimeError("DataLoader worker failed") from inner
    except RuntimeError as outer:
        assert is_fd_exhaustion(outer)
    assert not is_fd_exhaustion(ValueError("nope"))


def test_fd_pressure_is_relative_to_process_baseline(monkeypatch):
    monkeypatch.setattr(tuning, "_FD_BASELINE", None)
    monkeypatch.setattr(tuning, "fd_limits", lambda: (100, 100))
    counts = iter([60, 70, 81])
    monkeypatch.setattr(tuning, "open_fd_count", lambda: next(counts))
    assert tuning.fd_pressure(0.5) is None      # baseline 60 even though > 50% of limit
    assert tuning.fd_pressure(0.5) is None      # +10 of 40 headroom
    assert "grew from 60 to 81" in tuning.fd_pressure(0.5)


@pytest.mark.skipif(open_fd_count() is None, reason="needs /proc or /dev/fd")
def test_interrupted_fit_releases_loader_workers(tmp_path):
    """Regression: a pruned/NaN trial used to leave persistent workers alive."""
    torch.set_num_threads(1)
    cfg = {"smoke": True, "workers": 2, "batch_size": 2, "device": "cpu", "epochs": 3,
           "scheduler": "cosine", "max_grad_norm": 1.0, "objective": "vessel-f1", "min_score": 0.01,
           "threshold_min": 0.01, "threshold_max": 0.95, "match_distance": 20, "variant": "sar_only",
           "data_seed": 42, "max_train_samples": 0, "max_val_samples": 0, "deterministic": False}

    class Stop(Exception):
        pass

    def report(epoch, best):
        raise Stop()
    kept, before = [], open_fd_count()
    for index in range(3):
        with pytest.raises(Stop) as info:
            tuning.fit_one(cfg, "adamw", {"lr": 1e-4, "weight_decay": 0.0}, 42, tmp_path/str(index), report)
        kept.append(info.value)  # keep tracebacks alive, as Optuna's logging does
        assert not mp.active_children()
    assert open_fd_count() - before < 8
