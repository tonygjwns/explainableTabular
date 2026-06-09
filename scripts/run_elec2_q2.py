"""Q2b factorial: does STRUCTURE (time-TabR) beat time-as-a-FEATURE on measured concept?

Elec2 is the one dataset with a measured, exploitable concept (+0.132 within-overlap).
Q2 asks whether the time-TabR STRUCTURE exploits it better than a basis-matched
time-feature MLP. Factorial design (NEXT_TAB.md ★):

    {arch: mlp_t, tabr, time_tabr} x {time_basis: fourier, trend} x {split: temporal, random} x seed

Three arms share the SAME encoder + time basis (clean 'structure vs feature' / 'basis'):
  - mlp_t     : predictor([z ; basis(t)])         — time as a FEATURE (the baseline to beat)
  - tabr      : retrieval, time_mode=none         — structure, NO time (isolates 'time')
  - time_tabr : retrieval, time_mode=value        — structure + value-side label-drift hook

Pre-registered reads (PLAN_RESCUE / NEXT_TAB):
  - A structure gain must show on the TEMPORAL split (early->late stale labels) and be
    ~0 on RANDOM (train sees both periods). Gain only on random => red flag (not concept).
  - Decisive contrasts (paired Wilcoxon + Hedges' g over seeds):
      time_tabr(value) vs mlp_t  (basis-matched) : does structure carry time better?
      time_tabr        vs tabr                   : does adding the time hook help at all?

    python scripts/run_elec2_q2.py --config configs/phase1.yaml --n-seeds 25
    python scripts/run_elec2_q2.py --config configs/phase1.yaml --n-seeds 25 \
        --splits temporal --bases trend --time-mode value
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.utils.seed import seed_everything  # noqa: E402
from src.data.elec2_loader import load_elec2  # noqa: E402
from src.training.tabr_trainer import TabRConfig, train_timetabr  # noqa: E402
from src.utils.metrics import metric_name  # noqa: E402
from src.utils.stats import (  # noqa: E402
    orient_higher_is_better, paired_wilcoxon, hedges_g,
)


def _cfg(cfg, arch, time_mode, time_basis, seed, *, lr=None,
         min_epochs=0, record_history=False) -> TabRConfig:
    tr = cfg.training
    topk = OmegaConf.select(cfg, "tabr.topk", default=32)
    return TabRConfig(
        arch=arch, time_mode=time_mode, time_basis=time_basis,
        trend_degree=int(OmegaConf.select(cfg, "memory.trend_degree", default=3)),
        topk=int(topk),
        lr=float(lr) if lr is not None else float(tr.learning_rate),
        weight_decay=float(tr.weight_decay),
        batch_size=int(tr.batch_size), eval_batch=int(tr.eval_batch),
        patience=int(tr.patience), min_epochs=int(min_epochs),
        max_epochs=int(tr.max_epochs),
        record_history=record_history, seed_tag=f"s{seed}",
    )


def _diag(cfg, args):
    """Print the per-epoch val curve for one cell (temporal/trend/seed0, all arms).

    Disambiguates the 'epoch 1 is always best' observation: is it a real temporal-
    drift signature (val peaks early then degrades), or an optimization artifact that
    early-stops the zero-init time-modulation before it can train (-> time_tabr==tabr)?
    """
    split = args.splits[0]; basis = args.bases[0]
    print(f"\n==== DIAG: per-epoch val curve [{split}/{basis} s0], lr={args.lr or 'cfg'}, "
          f"min_epochs={args.min_epochs} ====")
    seed_everything(0)
    data = load_elec2(split=split, seed=0)
    for arch in args.archs:
        tm = args.time_mode if arch == "time_tabr" else "none"
        r = train_timetabr(data, _cfg(cfg, arch, tm, basis, 0, lr=args.lr,
                                      min_epochs=args.min_epochs, record_history=True))
        h = r["val_history"]
        curve = " ".join(f"{v:.4f}" for v in h[:25])
        print(f"  {arch:9s}/{tm:5s} best_epoch={r['best_epoch']:3d}/{r['n_epochs']:3d} "
              f"test={r['score']:.4f}\n      val: {curve}{' ...' if len(h) > 25 else ''}")
    print("  read: best_epoch>>0 with a rising-then-falling curve => mechanism gets to")
    print("        train; best_epoch~0 & monotone-down => artifact (raise min_epochs / lower lr).")


def _report_grid(cfg, args):
    """DECISION TABLE: multi-seed mean test per (arch, lr) + val->test rank corr.

    seed-0 diag can't conclude (noisy, and val<->test anti-correlate under concept
    drift => val-based selection is unreliable here). This reports MEAN TEST over
    n_seeds for EVERY (arch, lr) directly — no val selection — plus an oracle best-lr
    per arch (upper bound) and the val->test Spearman (negative => val misleads).
    One cell (splits[0]/bases[0]) so seeds buy power where it matters.

      python scripts/run_elec2_q2.py --config configs/phase1.yaml --report-grid \
          --n-seeds 10 --splits temporal --bases trend --lr-grid 2e-3 1e-3 5e-4 2e-4
    """
    from scipy.stats import spearmanr
    grid = args.lr_grid or [float(cfg.training.learning_rate)]
    split, basis = args.splits[0], args.bases[0]
    seeds = list(range(args.n_seeds))
    out_dir = Path(cfg.experiment.results_dir).parent / "elec2_q2"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cache: dict = {}
    def get_data(seed):
        key = seed if split == "random" else 0
        if key not in data_cache:
            data_cache[key] = load_elec2(split=split, seed=key)
        return data_cache[key]

    print(f"\n==== GRID REPORT [{split}/{basis}]  n_seeds={args.n_seeds}  grid={grid}  "
          f"min_epochs={args.min_epochs} ====")
    print(f"  {'arch':10s}{'lr':>9s}  {'mean_test':>9s} {'std':>6s}  {'mean_val':>8s}  best_epochs")
    cells = []
    for arch in args.archs:
        tm = args.time_mode if arch == "time_tabr" else "none"
        for lr in grid:
            tests, vals, bes = [], [], []
            for s in seeds:
                seed_everything(s)
                r = train_timetabr(get_data(s),
                                   _cfg(cfg, arch, tm, basis, s, lr=lr,
                                        min_epochs=args.min_epochs))
                tests.append(float(r["score"])); vals.append(float(r["val_score"]))
                bes.append(int(r["best_epoch"]))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            c = {"arch": arch, "time_mode": tm, "lr": lr,
                 "mean_test": float(np.mean(tests)), "std_test": float(np.std(tests)),
                 "mean_val": float(np.mean(vals)), "tests": tests, "vals": vals,
                 "best_epochs": bes}
            cells.append(c)
            print(f"  {arch:10s}{lr:9g}  {c['mean_test']:9.4f} {c['std_test']:6.4f}  "
                  f"{c['mean_val']:8.4f}  {bes}")

    print("\n  oracle best-TEST lr per arch (upper bound; NOT a val-fair number):")
    best = {}
    for arch in args.archs:
        ac = [c for c in cells if c["arch"] == arch]
        b = max(ac, key=lambda c: c["mean_test"])
        best[arch] = b
        print(f"    {arch:10s} lr={b['lr']:g}: test={b['mean_test']:.4f} ± {b['std_test']:.4f}")
    if {"mlp_t", "time_tabr"} <= set(args.archs):
        d_struct = best["time_tabr"]["mean_test"] - best["mlp_t"]["mean_test"]
        print(f"\n  DECISION (oracle): time_tabr - mlp_t = {d_struct:+.4f}  "
              f"(>~2*std => structure beats time-feature; ~0 => time helps but structure doesn't)")
    if {"tabr", "mlp_t"} <= set(args.archs):
        d_time = best["mlp_t"]["mean_test"] - best["tabr"]["mean_test"]
        print(f"            mlp_t    - tabr  = {d_time:+.4f}  (>0 => time itself helps over no-time retrieval)")

    mv = [c["mean_val"] for c in cells]; mt = [c["mean_test"] for c in cells]
    rho, p = spearmanr(mv, mt)
    print(f"\n  val->test Spearman across {len(cells)} cells: rho={rho:+.3f} (p={p:.3f})")
    print("    rho<=0 => val MISLEADS test (concept drift): do NOT select lr/epoch by val.")
    out = {"split": split, "basis": basis, "n_seeds": args.n_seeds, "grid": grid,
           "min_epochs": args.min_epochs, "cells": cells,
           "oracle_best_lr": {a: best[a]["lr"] for a in best},
           "val_test_spearman": float(rho)}
    (out_dir / "grid_report.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {out_dir / 'grid_report.json'}  <-- send me this one file")


def _contrast(scores_a, scores_b, metric):
    """Paired (time_tabr vs baseline) comparison, oriented so higher=better."""
    a = orient_higher_is_better(scores_a, metric)
    b = orient_higher_is_better(scores_b, metric)
    delta = float(np.mean(a) - np.mean(b))
    p = paired_wilcoxon(a, b)
    g = hedges_g(a, b)
    return {
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "delta_oriented": delta, "p_value": p, "hedges_g": g,
        "positive": bool(delta > 0 and p < 0.05 and g >= 0.5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--n-seeds", type=int, default=25,
                    help="seeds 0..n-1 (>=25 for power, or rely on g>=0.5)")
    ap.add_argument("--splits", nargs="+", default=["temporal", "random"],
                    choices=["temporal", "random"])
    ap.add_argument("--bases", nargs="+", default=["trend", "fourier"],
                    choices=["trend", "fourier"])
    ap.add_argument("--time-mode", default="value", choices=["value", "metric", "both"])
    ap.add_argument("--archs", nargs="+", default=["mlp_t", "tabr", "time_tabr"],
                    choices=["mlp_t", "tabr", "time_tabr"])
    ap.add_argument("--lr", type=float, default=None,
                    help="single learning rate override (else config). Ignored if --lr-grid given.")
    ap.add_argument("--lr-grid", nargs="+", type=float, default=None,
                    help="enable PER-(split,basis,arch) lr selection by val (fair comparison): "
                         "each arm runs at its own best-val lr from this grid. "
                         "e.g. --lr-grid 2e-3 1e-3 5e-4 2e-4")
    ap.add_argument("--tune-seeds", type=int, default=3,
                    help="#seeds (0..n-1) used to pick lr by mean val (no test peek)")
    ap.add_argument("--min-epochs", type=int, default=0,
                    help="floor before early-stop fires (let zero-init time-mod train)")
    ap.add_argument("--diag", action="store_true",
                    help="print per-epoch val curve for ONE cell, then exit (no full run)")
    ap.add_argument("--report-grid", action="store_true",
                    help="DECISION TABLE: multi-seed mean test per (arch,lr) + val/test "
                         "corr for ONE cell, then exit (use with --lr-grid --n-seeds)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.diag:
        _diag(cfg, args)
        return
    if args.report_grid:
        _report_grid(cfg, args)
        return
    seeds = list(range(args.n_seeds))
    out_dir = Path(cfg.experiment.results_dir).parent / "elec2_q2"
    out_dir.mkdir(parents=True, exist_ok=True)
    metric = metric_name("binclass")  # elec2 = roc_auc

    out_path = out_dir / f"q2_{args.time_mode}.json"
    print(f"Q2b factorial: archs={args.archs} bases={args.bases} splits={args.splits} "
          f"time_mode={args.time_mode} seeds=0..{args.n_seeds - 1}")
    print(f"results -> {out_path} (written incrementally after every seed)")

    # scores[split][basis][arch] = [per-seed test score]
    scores: dict = {sp: {b: {a: [] for a in args.archs} for b in args.bases} for sp in args.splits}
    rows = []
    summary = {"metric": metric, "n_seeds": args.n_seeds, "time_mode": args.time_mode,
               "splits": args.splits, "bases": args.bases,
               "lr": args.lr, "lr_grid": args.lr_grid, "tune_seeds": args.tune_seeds,
               "min_epochs": args.min_epochs, "lr_selected": {}, "complete": False,
               "contrasts": [], "rows": rows}
    if args.lr_grid:
        print(f"per-arm lr selection ON: grid={args.lr_grid}, tune on "
              f"{min(args.tune_seeds, args.n_seeds)} seed(s) by mean val (no test peek)")

    # ---- memoized single trainings: each (split,basis,arch,seed,lr) trained once,
    #      so lr-selection tune-seeds are reused by the main sweep (no duplicate work).
    data_cache: dict = {}
    result_cache: dict = {}

    def get_data(split, seed):
        # temporal split is seed-independent (idx=arange); cache it once.
        key = (split, seed if split == "random" else 0)
        if key not in data_cache:
            data_cache[key] = load_elec2(split=split, seed=key[1])
        return data_cache[key]

    def run_one(split, basis, arch, seed, lr):
        tm = args.time_mode if arch == "time_tabr" else "none"
        key = (split, basis, arch, seed, lr)
        if key not in result_cache:
            seed_everything(seed)
            r = train_timetabr(get_data(split, seed),
                               _cfg(cfg, arch, tm, basis, seed, lr=lr,
                                    min_epochs=args.min_epochs))
            result_cache[key] = {"score": float(r["score"]), "val_score": float(r["val_score"]),
                                 "best_epoch": int(r["best_epoch"]), "n_epochs": int(r["n_epochs"]),
                                 "time_mode": tm}
            del r
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return result_cache[key]

    def select_lr(split, basis, arch):
        """Pick the lr maximizing mean VAL over tune seeds (test never inspected)."""
        if not args.lr_grid:
            return args.lr   # None => config default; single shared lr (old behavior)
        tune = list(range(min(args.tune_seeds, args.n_seeds)))
        best_lr, best_val = None, -np.inf
        for lr in args.lr_grid:
            v = float(np.mean([run_one(split, basis, arch, s, lr)["val_score"] for s in tune]))
            if v > best_val:
                best_val, best_lr = v, lr
        print(f"  [lr-select {split:8s}/{basis:7s}/{arch:9s}] -> lr={best_lr:g} (mean val={best_val:.4f})")
        return best_lr

    def save():
        out_path.write_text(json.dumps(summary, indent=2))

    for split in args.splits:
        for basis in args.bases:
            lr_by_arch = {a: select_lr(split, basis, a) for a in args.archs}
            summary["lr_selected"][f"{split}/{basis}"] = {
                a: lr_by_arch[a] for a in args.archs}
            for seed in seeds:
                line = {}
                for arch in args.archs:
                    lr = lr_by_arch[arch]
                    r = run_one(split, basis, arch, seed, lr)
                    s = r["score"]
                    scores[split][basis][arch].append(s)
                    line[arch] = (s, r["best_epoch"], r["n_epochs"])
                    rows.append({"split": split, "basis": basis, "arch": arch,
                                 "time_mode": r["time_mode"], "seed": seed, "score": s,
                                 "lr": lr, "val_score": r["val_score"],
                                 "best_epoch": r["best_epoch"], "n_epochs": r["n_epochs"]})
                ldesc = "  ".join(f"{a}={line[a][0]:.4f}@e{line[a][1]}/{line[a][2]}"
                                  for a in args.archs)
                print(f"[{split:8s}/{basis:7s} s{seed:02d}] {ldesc}")
                save()    # incremental: crash mid-run still leaves all finished rows

    # ---- pre-registered contrasts ----
    summary["complete"] = True
    print("\n==== Q2b contrasts (time_tabr vs baseline; +g => structure wins) ====")
    for split in args.splits:
        for basis in args.bases:
            sc = scores[split][basis]
            if "time_tabr" not in sc:
                continue
            for base in ("mlp_t", "tabr"):
                if base not in sc:
                    continue
                c = _contrast(sc["time_tabr"], sc[base], metric)
                c.update({"split": split, "basis": basis,
                          "candidate": "time_tabr", "baseline": base})
                summary["contrasts"].append(c)
                print(f"  [{split:8s}/{basis:7s}] time_tabr vs {base:6s}: "
                      f"delta={c['delta_oriented']:+.4f}  p={c['p_value']:.4f}  "
                      f"g={c['hedges_g']:+.2f}  positive={c['positive']}")

    # pre-registered sanity: structure gain should localize to temporal, ~0 on random
    if {"temporal", "random"} <= set(args.splits):
        print("\n  (pre-registered: gain on TEMPORAL, ~0 on RANDOM => concept exploit;")
        print("                   gain only on RANDOM => red flag, not concept.)")

    save()    # final write: now with contrasts + complete=True
    print(f"\nwrote {out_path}  <-- send me THIS one file (has all rows + contrasts)")


if __name__ == "__main__":
    main()
