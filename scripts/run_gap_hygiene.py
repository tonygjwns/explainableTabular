"""V3.3 hygiene — one decisive pass that hardens the within-overlap concept gap
before the advisor alignment (PLAN_V3 §F2 + §V3.3). Bundles the five hygiene items
the rebuild still owes, all sharing the SAME overlap-band machinery and the SAME real
concept datasets (elec2, insects), so a single server run discharges them together:

  (3) SEED-CI on every real gap: N-seed mean + 95% CI (true and placebo).
  (4) LOSS-ROBUSTNESS: the gap re-measured under Brier, log-loss (= Bayes-risk under
      log loss) and KL (rule-movement) besides AUC/accuracy. The concept VERDICT must
      not hinge on the metric (recalibration-drift blind spot, D2 ②③).
  (5) ROLLING-ORIGIN g(t): gap at a grid of time cut-points (not just the median) ->
      shape stats (trend Spearman, abrupt max-jump) + cross-cut CI (= replication, R7).
  (1) BH-FDR over the CONTRAST FAMILY: one one-sided test per dataset (true_gap >
      placebo, paired by seed), Benjamini-Hochberg across the family (R12).
  (2) CLAIM-A PRE-REGISTERED THRESHOLD + SENSITIVITY GRID: a verdict rule fixed up
      front (CI above placebo AND bias-corrected gap > noise floor AND BH-significant
      AND metric-invariant), then a band x min_per_half x classifier grid showing the
      verdict is invariant to those arbitrary choices (R11).

Pre-registered decision rule (committed here, BEFORE looking at the numbers):
  Claim "genuine concept on the deployed representation" for a dataset IFF
    (a) true-gap 95% CI lower bound > placebo 95% CI upper bound, AND
    (b) bias-corrected gap (true - placebo) > --floor   (default 0.034 = the G1
        noise-drift null, so the gap must clear the documented Bayes-noise confound), AND
    (c) BH-adjusted one-sided p (true > placebo) < --alpha, AND
    (d) metric-invariant: every proper-scoring gap (auc/accuracy, brier, logloss) is
        positive (late better) at the default band.
  Else -> PLAN_V3 §0.1 honest branch for that dataset.

Model-light (sklearn HGB/logreg). Run on the server (real data); --synth-only runs
anywhere as a wiring smoke test.

    python scripts/run_gap_hygiene.py --elec2 --insects
    python scripts/run_gap_hygiene.py --elec2 --insects --tabred cooking_time maps_routing
    python scripts/run_gap_hygiene.py --synth-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon, spearmanr  # noqa: E402

from src.analysis.drift_measure import _stack, concept_within_overlap_multi  # noqa: E402
# benjamini_hochberg imported lazily in main() (pulls statsmodels; only needed for the
# real-data BH-FDR section, so --synth-only wiring works without it).

PROPER = ("auc", "accuracy", "brier", "logloss", "rmse", "mae")  # signed skill gaps


def _ci95(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return (float("nan"), float("nan"))
    m, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
    return (float(m - 1.96 * se), float(m + 1.96 * se))


def _measure(Xe, ye, Xl, yl, task, seed, permute, band, min_per_half, clf):
    r = concept_within_overlap_multi(Xe, ye, Xl, yl, task, seed=seed, permute_time=permute,
                                     band=band, min_per_half=min_per_half, clf=clf)
    return r.get("gaps") if r.get("measurable") else None


# ---------------------------------------------------------------- (3)+(4) seed-CI + loss-robust
def seed_metrics(name, Xe, ye, Xl, yl, task, n_seeds, band, min_per_half, floor):
    """Per-metric seed distribution (true & placebo) at the default band/cut."""
    true, plac = [], []
    for s in range(n_seeds):
        gt = _measure(Xe, ye, Xl, yl, task, s, False, band, min_per_half, "hgb")
        gp = _measure(Xe, ye, Xl, yl, task, s, True, band, min_per_half, "hgb")
        if gt is not None:
            true.append(gt)
        if gp is not None:
            plac.append(gp)
    if not true:
        return {"dataset": name, "task": task, "measurable": False}
    metrics = list(true[0].keys())
    primary = "rmse" if task == "regression" else ("auc" if task == "binclass" else "accuracy")
    per_metric = {}
    for mname in metrics:
        tv = np.array([d[mname] for d in true if mname in d])
        pv = np.array([d[mname] for d in plac if mname in d]) if plac else np.array([])
        t_ci = _ci95(tv); p_ci = _ci95(pv) if len(pv) else (float("nan"), float("nan"))
        bias = float(tv.mean() - pv.mean()) if len(pv) else float("nan")
        per_metric[mname] = {
            "true_mean": float(tv.mean()), "true_ci": t_ci,
            "placebo_mean": float(pv.mean()) if len(pv) else None, "placebo_ci": p_ci,
            "bias_corrected": bias,
            "ci_above_placebo": bool(len(pv) and t_ci[0] > p_ci[1]),
            "ci_excludes_0": bool(t_ci[0] > 0),
        }
    # one-sided paired Wilcoxon (true > placebo) on the PRIMARY metric -> family p
    tprim = np.array([d[primary] for d in true if primary in d])
    pprim = np.array([d[primary] for d in plac if primary in d]) if plac else np.array([])
    pval = float("nan")
    if len(tprim) == len(pprim) and len(tprim) >= 2 and not np.allclose(tprim - pprim, 0):
        try:
            pval = float(wilcoxon(tprim, pprim, alternative="greater").pvalue)
        except ValueError:
            pval = float("nan")
    # metric-invariance: every proper-scoring gap positive at the default band
    proper = [m for m in metrics if m in PROPER]
    metric_invariant = bool(proper and all(per_metric[m]["true_mean"] > 0 for m in proper))
    return {"dataset": name, "task": task, "measurable": True, "primary_metric": primary,
            "n_seeds_used": len(true), "per_metric": per_metric,
            "primary_one_sided_p": pval, "metric_invariant": metric_invariant,
            "floor": floor}


# ---------------------------------------------------------------- (5) rolling-origin g(t)
def rolling_origin(name, X, y, t, task, cuts, n_seeds, band, min_per_half):
    primary_traj = []
    for q in cuts:
        thr = float(np.quantile(t, q))
        em, lm = t <= thr, t > thr
        if em.sum() < 300 or lm.sum() < 300:
            primary_traj.append(None); continue
        gs = []
        for s in range(n_seeds):
            g = _measure(X[em], y[em], X[lm], y[lm], task, s, False, band, min_per_half, "hgb")
            if g is not None:
                gs.append(g[("rmse" if task == "regression"
                             else "auc" if task == "binclass" else "accuracy")])
        primary_traj.append(float(np.mean(gs)) if gs else None)
    vals = [(q, g) for q, g in zip(cuts, primary_traj) if g is not None]
    shape = {}
    if len(vals) >= 3:
        qs = np.array([q for q, _ in vals]); gv = np.array([g for _, g in vals])
        rho, pp = spearmanr(qs, gv)
        jumps = np.abs(np.diff(gv))
        shape = {"trend_spearman": float(rho), "trend_p": float(pp),
                 "max_adjacent_jump": float(jumps.max()),
                 "cross_cut_mean": float(gv.mean()), "cross_cut_ci": _ci95(gv),
                 "all_positive": bool(np.all(gv > 0)),
                 "regime": ("abrupt" if jumps.max() > 2 * np.median(jumps) + 1e-9
                            else "gradual")}
    return {"dataset": name, "cuts": list(cuts), "primary_metric":
            ("rmse" if task == "regression" else "auc" if task == "binclass" else "accuracy"),
            "trajectory": primary_traj, "shape": shape}


# ---------------------------------------------------------------- (2) sensitivity grid
def sensitivity_grid(name, Xe, ye, Xl, yl, task, n_seeds, bands, mphs, clfs, floor):
    primary = "rmse" if task == "regression" else ("auc" if task == "binclass" else "accuracy")
    cells = []
    for band in bands:
        for mph in mphs:
            for clf in clfs:
                tg, pg = [], []
                for s in range(n_seeds):
                    gt = _measure(Xe, ye, Xl, yl, task, s, False, band, mph, clf)
                    gp = _measure(Xe, ye, Xl, yl, task, s, True, band, mph, clf)
                    if gt is not None:
                        tg.append(gt[primary])
                    if gp is not None:
                        pg.append(gp[primary])
                if not tg:
                    cells.append({"band": list(band), "min_per_half": mph, "clf": clf,
                                  "measurable": False}); continue
                t_ci = _ci95(tg); p_ci = _ci95(pg) if pg else (float("nan"), float("nan"))
                bias = float(np.mean(tg) - np.mean(pg)) if pg else float("nan")
                verdict = bool(pg and t_ci[0] > p_ci[1] and bias > floor)
                cells.append({"band": list(band), "min_per_half": mph, "clf": clf,
                              "measurable": True, "true_mean": float(np.mean(tg)),
                              "placebo_mean": float(np.mean(pg)) if pg else None,
                              "bias_corrected": bias, "verdict_concept": verdict})
    meas = [c for c in cells if c.get("measurable")]
    n_concept = sum(c["verdict_concept"] for c in meas)
    return {"dataset": name, "n_cells": len(meas), "n_verdict_concept": n_concept,
            "verdict_invariant": bool(meas and n_concept == len(meas)),
            "cells": cells}


# ---------------------------------------------------------------- synthetic smoke (wiring)
def _synth(seed, flip):
    rng = np.random.default_rng(seed)
    Xe = rng.normal(0, 1, (6000, 6)); Xl = rng.normal(0, 1, (6000, 6))
    ye = (3.0 * Xe[:, 0] + rng.normal(0, 0.4, 6000) > 0).astype(int)
    coef = -3.0 if flip else 3.0
    yl = (coef * Xl[:, 0] + rng.normal(0, 0.4, 6000) > 0).astype(int)
    return Xe, ye, Xl, yl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--elec2", action="store_true")
    ap.add_argument("--insects", action="store_true")
    ap.add_argument("--insects-variant", default="incremental_balanced")
    ap.add_argument("--tabred", nargs="*", default=[], help="TabReD dataset names (controls)")
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--grid-seeds", type=int, default=5, help="seeds per sensitivity/cut cell")
    ap.add_argument("--floor", type=float, default=0.034,
                    help="pre-registered noise floor (G1 noise-drift null)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--band", type=float, nargs=2, default=(0.1, 0.9))
    ap.add_argument("--min-per-half", type=int, default=200)
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--synth-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path("results/phase1/gap_hygiene"); out_dir.mkdir(parents=True, exist_ok=True)
    band = tuple(args.band)
    bands = [(0.1, 0.9), (0.2, 0.8), (0.05, 0.95)]
    mphs = [args.min_per_half, max(100, args.min_per_half // 2), args.min_per_half * 2]
    clfs = ["hgb", "logreg"]

    if args.synth_only:
        print("\n==== SYNTHETIC wiring smoke (rule flips => concept; no-flip => ~0) ====")
        for tag, flip in [("flip(+ctrl)", True), ("noflip(null)", False)]:
            Xe, ye, Xl, yl = _synth(0, flip)
            r = seed_metrics(tag, Xe, ye, Xl, yl, "binclass", min(args.n_seeds, 5),
                             band, args.min_per_half, args.floor)
            pm = r["per_metric"]
            print(f"  {tag:14s} auc={pm['auc']['true_mean']:+.3f} "
                  f"brier={pm['brier']['true_mean']:+.3f} logloss={pm['logloss']['true_mean']:+.3f} "
                  f"kl={pm['kl_late_early']['true_mean']:.3f} invariant={r['metric_invariant']}")
        print("  (flip must be large+invariant; noflip ~0) -> wiring OK")
        return

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config); root = Path(cfg.data.root)
    jobs = []
    if args.elec2:
        from src.data.elec2_loader import load_elec2
        jobs.append(("elec2", load_elec2(split="temporal", seed=0)))
    if args.insects:
        from src.data.insects_loader import load_insects
        jobs.append((f"insects_{args.insects_variant}",
                     load_insects(variant=args.insects_variant, split="temporal", seed=0)))
    for ds in args.tabred:
        from src.data.tabred_loader import load_tabred
        jobs.append((ds, load_tabred(ds, root, split=cfg.experiment.split)))

    report = {"floor": args.floor, "alpha": args.alpha, "band": list(band),
              "datasets": [], "rolling": [], "sensitivity": []}
    family_p, family_names = [], []

    print(f"\n==== (3)+(4) seed-CI + loss-robustness ({args.n_seeds} seeds, band={band}) ====")
    for name, data in jobs:
        t = data.train.t; med = float(np.median(t))
        X = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
        em, lm = t <= med, t > med
        r = seed_metrics(name, X[em], data.train.y[em], X[lm], data.train.y[lm],
                         data.task, args.n_seeds, band, args.min_per_half, args.floor)
        report["datasets"].append(r)
        if not r.get("measurable"):
            print(f"  {name:24s} NOT MEASURABLE"); continue
        pm = r["per_metric"]; prim = r["primary_metric"]
        family_p.append(r["primary_one_sided_p"]); family_names.append(name)
        print(f"  {name} [{prim}]:")
        for mname, v in pm.items():
            tci = v["true_ci"]
            extra = (f"  bias_corr={v['bias_corrected']:+.4f}" if v["placebo_mean"] is not None
                     else "")
            print(f"    {mname:14s} true={v['true_mean']:+.4f}[{tci[0]:+.4f},{tci[1]:+.4f}]"
                  f"{extra}")
        print(f"    one-sided p(true>placebo)={r['primary_one_sided_p']:.4g}  "
              f"metric_invariant={r['metric_invariant']}")

    # ---- (1) BH-FDR over the contrast family ----
    print(f"\n==== (1) BH-FDR over contrast family (n={len(family_p)}, alpha={args.alpha}) ====")
    bh = {}
    if family_p:
        from src.utils.stats import benjamini_hochberg
        valid = [(n, p) for n, p in zip(family_names, family_p) if not np.isnan(p)]
        if valid:
            names_v, ps_v = zip(*valid)
            reject, p_corr = benjamini_hochberg(ps_v, alpha=args.alpha)
            for n, p, pc, rj in zip(names_v, ps_v, p_corr, reject):
                bh[n] = {"raw_p": float(p), "bh_p": float(pc), "reject": bool(rj)}
                print(f"  {n:24s} raw_p={p:.4g}  BH_p={pc:.4g}  reject={bool(rj)}")
    report["bh_fdr"] = bh

    # ---- (5) rolling-origin g(t) ----
    print(f"\n==== (5) rolling-origin g(t) over cuts {args.cuts} ({args.grid_seeds} seeds/cut) ====")
    for name, data in jobs:
        t = data.train.t
        X = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
        r = rolling_origin(name, X, data.train.y, t, data.task, args.cuts,
                           args.grid_seeds, band, args.min_per_half)
        report["rolling"].append(r)
        traj = "  ".join(f"{q}:{g:+.3f}" if g is not None else f"{q}:--"
                         for q, g in zip(args.cuts, r["trajectory"]))
        sh = r["shape"]
        print(f"  {name:24s} {traj}")
        if sh:
            print(f"    trend_rho={sh['trend_spearman']:+.2f} regime={sh['regime']} "
                  f"cross-cut CI=[{sh['cross_cut_ci'][0]:+.3f},{sh['cross_cut_ci'][1]:+.3f}] "
                  f"all_pos={sh['all_positive']}")

    # ---- (2) sensitivity grid ----
    print(f"\n==== (2) sensitivity grid band x min_per_half x clf ({args.grid_seeds} seeds) ====")
    for name, data in jobs:
        t = data.train.t; med = float(np.median(t))
        X = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
        em, lm = t <= med, t > med
        r = sensitivity_grid(name, X[em], data.train.y[em], X[lm], data.train.y[lm],
                             data.task, args.grid_seeds, bands, mphs, clfs, args.floor)
        report["sensitivity"].append(r)
        print(f"  {name:24s} verdict-concept in {r['n_verdict_concept']}/{r['n_cells']} cells "
              f"-> invariant={r['verdict_invariant']}")

    # ---- pre-registered Claim-A verdict ----
    print("\n==== PRE-REGISTERED Claim-A verdict (per dataset) ====")
    verdicts = {}
    sens_by = {r["dataset"]: r for r in report["sensitivity"]}
    for r in report["datasets"]:
        if not r.get("measurable"):
            verdicts[r["dataset"]] = {"concept": False, "reason": "not measurable"}; continue
        name = r["dataset"]; prim = r["primary_metric"]; v = r["per_metric"][prim]
        a = v["ci_above_placebo"]
        b = (v["bias_corrected"] is not None and not np.isnan(v["bias_corrected"])
             and v["bias_corrected"] > args.floor)
        c = bool(bh.get(name, {}).get("reject", False))
        d = r["metric_invariant"]
        sg = sens_by.get(name, {}).get("verdict_invariant", None)
        concept = bool(a and b and c and d)
        verdicts[name] = {"concept": concept, "ci_above_placebo": a,
                          "bias_above_floor": b, "bh_reject": c, "metric_invariant": d,
                          "sensitivity_invariant": sg}
        print(f"  {name:24s} CI>placebo={a} bias>{args.floor}={b} BH={c} metric_inv={d} "
              f"sens_inv={sg}  =>  {'CONCEPT' if concept else 'NOT (PLAN_V3 §0.1)'}")
    report["claim_a_verdict"] = verdicts

    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n  wrote {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
