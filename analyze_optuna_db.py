#!/usr/bin/env python3
"""Summarize an Optuna database produced by tuning.py (v1 per-optimizer or v2 single studies).

Reads the SQLite file directly (no Optuna import), so it also works on a copy of
the database while a search is running. Prints, per study:
  * each trial's optimizer, lr, weight decay, value and the epoch of its best score;
  * per-optimizer best / median;
  * replicate noise: identical (variant, optimizer, lr, wd, seeds) configs that
    were run in more than one study with the same code hash.

    python analyze_optuna_db.py tuning_runs/optuna.db [--prefix full-v1] [--json out.json]
"""
import argparse
from collections import defaultdict
import json
import sqlite3
import statistics


def decode(conn):
    studies = {}
    for study_id, name in conn.execute("SELECT study_id, study_name FROM studies"):
        attrs = {k: json.loads(v) for k, v in conn.execute(
            "SELECT key, value_json FROM study_user_attributes WHERE study_id=?", (study_id,))}
        studies[study_id] = {"name": name, "contract": attrs.get("contract", {}), "trials": []}
    for trial_id, study_id, number, state in conn.execute(
            "SELECT trial_id, study_id, number, state FROM trials ORDER BY study_id, number"):
        params = {}
        for key, value, dist in conn.execute(
                "SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=?", (trial_id,)):
            dist = json.loads(dist)
            params[key] = (dist["attributes"]["choices"][int(value)]
                           if dist["name"] == "CategoricalDistribution" else value)
        value = conn.execute("SELECT value FROM trial_values WHERE trial_id=?", (trial_id,)).fetchone()
        curve = [v for (v,) in conn.execute("SELECT intermediate_value FROM trial_intermediate_values "
                                            "WHERE trial_id=? ORDER BY step", (trial_id,))]
        user = {k: json.loads(v) for k, v in conn.execute(
            "SELECT key, value_json FROM trial_user_attributes WHERE trial_id=?", (trial_id,))}
        study = studies[study_id]
        contract = study["contract"]
        name = params.get("optimizer") or contract.get("optimizer_name") or study["name"].split("__")[-1]
        lr = params.get(f"{name}_lr", params.get("lr"))
        wd = params.get(f"{name}_weight_decay", params.get("weight_decay"))
        study["trials"].append({"number": number, "state": state, "optimizer": name, "lr": lr,
                                "weight_decay": wd, "value": value[0] if value else None,
                                "best_epoch": curve.index(max(curve)) + 1 if curve else None,
                                "seed_scores": user.get("seed_scores"),
                                "failure_reason": user.get("failure_reason")})
    return list(studies.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("database")
    parser.add_argument("--prefix", help="Only studies whose name starts with this prefix")
    parser.add_argument("--json", help="Also write the decoded trials to this JSON file")
    args = parser.parse_args()
    conn = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    studies = [s for s in decode(conn) if not args.prefix or s["name"].startswith(args.prefix)]
    replicates = defaultdict(list)
    for study in studies:
        config = study["contract"].get("config", {})
        print(f"\n=== {study['name']}  (epochs={config.get('epochs')}, seeds={study['contract'].get('seeds')}, "
              f"smoke={config.get('smoke')})")
        by_opt = defaultdict(list)
        for t in study["trials"]:
            lr = f"{t['lr']:.2e}" if t["lr"] is not None else "-"
            value = f"{t['value']:.4f}" if t["value"] is not None else "-"
            extra = f" FAIL: {t['failure_reason']}" if t["failure_reason"] else ""
            print(f"  #{t['number']:<3} {t['state']:<8} {t['optimizer']:<8} lr={lr:<9} wd={t['weight_decay']!s:<7} "
                  f"value={value:<7} best@epoch={t['best_epoch']}{extra}")
            if t["value"] is not None:
                by_opt[t["optimizer"]].append(t["value"])
                if config and not config.get("smoke"):
                    key = (config.get("code_sha256"), config.get("variant"), t["optimizer"],
                           round(t["lr"], 12), t["weight_decay"], tuple(study["contract"].get("seeds", [])))
                    replicates[key].append((study["name"], t["value"]))
        for name, values in by_opt.items():
            print(f"  -> {name:<8} n={len(values):<3} best={max(values):.4f} median={statistics.median(values):.4f}")
    pairs = {k: v for k, v in replicates.items() if len({s for s, _ in v}) > 1}
    if pairs:
        print("\n=== Replicates: same code, variant, optimizer, lr, wd and seeds in different studies")
        spreads = []
        for (_, variant, name, lr, wd, _), runs in pairs.items():
            values = [v for _, v in runs]
            spreads.append(max(values) - min(values))
            print(f"  {variant}/{name} lr={lr:.2e} wd={wd}: " + ", ".join(f"{v:.4f}" for v in values)
                  + f"  (spread {spreads[-1]:.4f})")
        print(f"  median spread {statistics.median(spreads):.4f}; max {max(spreads):.4f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(studies, f, indent=1)


if __name__ == "__main__":
    main()
