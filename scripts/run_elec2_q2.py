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
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.utils.seed import seed_everything  # noqa: E402
from src.data.elec2_loader import load_elec2  # noqa: E402
from src.data.insects_loader import load_insects  # noqa: E402
from src.training.tabr_trainer import TabRConfig, train_timetabr  # noqa: E402
from src.utils.metrics import metric_name  # noqa: E402
from src.utils.stats import (  # noqa: E402
    orient_higher_is_better, paired_wilcoxon, hedges_g,
)


DIAG_FILE = "diagnostics.jsonl"   # all --diag / --report-grid runs append here (1 line each)


def _params(args):
    """The run knobs, stamped on every appended record so runs are distinguishable."""
    return {"splits": args.splits, "bases": args.bases, "archs": args.archs,
            "time_mode": args.time_mode, "lr": args.lr, "lr_grid": args.lr_grid,
            "batch_size": args.batch_size, "weight_decay": args.weight_decay,
            "dropout": args.dropout, "eval_every_steps": args.eval_every_steps,
            "min_epochs": args.min_epochs, "n_seeds": args.n_seeds}


def _append(out_dir, record):
    """Append one JSON record as a line to diagnostics.jsonl (accumulates all runs)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    record["ts"] = time.time()
    with open(out_dir / DIAG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return out_dir / DIAG_FILE


def _load(args, split, seed):
    """Dispatch the dataset (elec2 binclass | insects multiclass)."""
    if args.dataset == "insects":
        return load_insects(variant=args.insects_variant, split=split, seed=seed,
                            max_samples=args.max_samples)
    return load_elec2(split=split, seed=seed)


def _cfg(cfg, arch, time_mode, time_basis, seed, *, lr=None,
         min_epochs=0, record_history=False, args=None) -> TabRConfig:
    tr = cfg.training
    topk = OmegaConf.select(cfg, "tabr.topk", default=32)
    # optional overrides (diagnostics ②③④): batch / weight_decay / dropout / step-eval
    bs = args.batch_size if args and args.batch_size else int(tr.batch_size)
    wd = args.weight_decay if args and args.weight_decay is not None else float(tr.weight_decay)
    do = args.dropout if args and args.dropout is not None else 0.0
    evs = args.eval_every_steps if args and args.eval_every_steps else 0
    return TabRConfig(
        arch=arch, time_mode=time_mode, time_basis=time_basis,
        trend_degree=int(OmegaConf.select(cfg, "memory.trend_degree", default=3)),
        topk=int(topk),
        lr=float(lr) if lr is not None else float(tr.learning_rate),
        weight_decay=float(wd), dropout=float(do),
        batch_size=int(bs), eval_batch=int(tr.eval_batch),
        patience=int(tr.patience), min_epochs=int(min_epochs),
        eval_every_steps=int(evs), max_epochs=int(tr.max_epochs),
        record_history=record_history, seed_tag=f"s{seed}",
    )


def _diag(cfg, args):
    """① temporal-vs-random learning curves (train loss + val) — bug vs drift decider.

    Runs each requested split, all arms, seed 0, recording per-epoch TRAIN LOSS and VAL.
    Reading (per the diagnostic plan):
      - train loss falling but val peak-then-decline, AND temporal peaks early while
        random peaks later/flat  => real concept DRIFT (not a bug). expected.
      - train loss flat           => optimization bug (no gradient flow).
      - BOTH splits peak at epoch 0/1 & val flat+noisy => saturation / val too small.
    Use --eval-every-steps N to see sub-epoch resolution; --dropout/--weight-decay to
    flatten early overfit so the mechanism gets to train.
    """
    out_dir = Path(cfg.experiment.results_dir).parent / f"{args.dataset}_q2"
    print(f"\n==== DIAG curves: splits={args.splits} basis={args.bases[0]} s0  "
          f"lr={args.lr or 'cfg'} bs={args.batch_size or 'cfg'} wd={args.weight_decay} "
          f"dropout={args.dropout} eval_every_steps={args.eval_every_steps} "
          f"min_epochs={args.min_epochs} ====")
    basis = args.bases[0]
    curves = []
    for split in args.splits:
        print(f"\n  --- split={split} ---")
        seed_everything(0)
        data = _load(args, split, 0)
        for arch in args.archs:
            tm = args.time_mode if arch == "time_tabr" else "none"
            r = train_timetabr(data, _cfg(cfg, arch, tm, basis, 0, lr=args.lr,
                                          min_epochs=args.min_epochs,
                                          record_history=True, args=args))
            vh, lh = r["val_history"], r["train_loss_history"]
            unit = "step-evals" if args.eval_every_steps else "epochs"
            vcurve = " ".join(f"{v:.4f}" for v in vh[:30])
            lcurve = " ".join(f"{v:.3f}" for v in lh[:30])
            print(f"  {arch:9s}/{tm:5s} best@{r['best_epoch']}/{r['n_epochs']}ep "
                  f"test={r['score']:.4f}  argmax_val={int(np.argmax(vh))}/{len(vh)-1} {unit}")
            print(f"      train_loss: {lcurve}{' ...' if len(lh) > 30 else ''}")
            print(f"      val       : {vcurve}{' ...' if len(vh) > 30 else ''}")
            curves.append({"split": split, "arch": arch, "time_mode": tm,
                           "test": float(r["score"]), "val_score": float(r["val_score"]),
                           "best_epoch": int(r["best_epoch"]), "n_epochs": int(r["n_epochs"]),
                           "argmax_val": int(np.argmax(vh)), "val_unit": unit,
                           "val_history": vh, "train_loss_history": lh})
    print("\n  VERDICT: temporal peaks early + random peaks later/flat => DRIFT (not bug).")
    print("           train_loss must DECREASE (else gradient bug). both-splits-flat-val => saturation.")
    p = _append(out_dir, {"mode": "diag", "params": _params(args), "curves": curves})
    print(f"\n  appended to {p}  <-- send me THIS file (accumulates every diag/grid run)")


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
    out_dir = Path(cfg.experiment.results_dir).parent / f"{args.dataset}_q2"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cache: dict = {}
    def get_data(seed):
        key = seed if split == "random" else 0
        if key not in data_cache:
            data_cache[key] = _load(args, split, key)
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
                                        min_epochs=args.min_epochs, args=args))
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
    record = {"mode": "report_grid", "params": _params(args),
              "split": split, "basis": basis, "grid": grid, "cells": cells,
              "oracle_best_lr": {a: best[a]["lr"] for a in best},
              "val_test_spearman": float(rho)}
    p = _append(out_dir, record)
    print(f"\n  appended to {p}  <-- send me THIS file (accumulates every diag/grid run)")


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
    ap.add_argument("--dataset", default="elec2", choices=["elec2", "insects"],
                    help="elec2 (binclass) | insects (multiclass designed-drift, 2nd dataset)")
    ap.add_argument("--insects-variant", default="incremental_balanced",
                    help="river INSECTS variant (see insects_loader.VARIANTS)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap INSECTS stream length (head) for speed; None=full")
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
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override batch size (smaller => more updates/epoch, sees peak)")
    ap.add_argument("--weight-decay", type=float, default=None,
                    help="override weight decay (regularize early overfit; applied to all arms)")
    ap.add_argument("--dropout", type=float, default=None,
                    help="encoder dropout (regularize; applied to all arms)")
    ap.add_argument("--eval-every-steps", type=int, default=0,
                    help="eval every N optimizer steps (sub-epoch resolution); 0=per-epoch")
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
    out_dir = Path(cfg.experiment.results_dir).parent / f"{args.dataset}_q2"
    out_dir.mkdir(parents=True, exist_ok=True)
    metric = metric_name("multiclass" if args.dataset == "insects" else "binclass")

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
               "min_epochs": args.min_epochs, "batch_size": args.batch_size,
               "weight_decay": args.weight_decay, "dropout": args.dropout,
               "eval_every_steps": args.eval_every_steps,
               "lr_selected": {}, "complete": False, "contrasts": [], "rows": rows}
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
            data_cache[key] = _load(args, split, key[1])
        return data_cache[key]

    def run_one(split, basis, arch, seed, lr):
        tm = args.time_mode if arch == "time_tabr" else "none"
        key = (split, basis, arch, seed, lr)
        if key not in result_cache:
            seed_everything(seed)
            r = train_timetabr(get_data(split, seed),
                               _cfg(cfg, arch, tm, basis, seed, lr=lr,
                                    min_epochs=args.min_epochs, args=args))
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
