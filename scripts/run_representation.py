"""V3.0 G2 — is "disjoint support / unmeasurable" the DATA, or feature engineering? (PLAN_V3 §G2)

Red-team D1 (deepest): "covariate dominance ⇒ concept unmeasurable" is NOT
representation-invariant. Adding any time-proxy feature (lag, expanding-window,
calendar — what TabReD's 261-feature pipeline produces by default) drives cov_AUC→1
and overlap→0 REGARDLESS of whether P(y|x) moved. So "unmeasurable on TabReD" may be
"unmeasurable in TabReD's chosen 261-feature representation." This script tests it
both directions and DECIDES whether Claim A must be re-scoped to the deployed
representation (it should be, either way the result is reportable).

Per dataset, recompute (cov_AUC, overlap_mass, concept gap) under representations:
  - full          : all features (baseline = current §13 verdict)
  - drop_timeproxy: drop features with high |corr(feature, t)| (the engineered time
                    leakage); keep the rest. Does overlap come back on disjoint sets?
  - sparse_MI@k   : keep the top-k features by mutual information with y (predictive,
                    NOT time-predictive), k ∈ {5,10,20,50}. A sane practitioner's
                    representation. Does concept become measurable?
  - add_timeproxy : (for MEASURABLE datasets) APPEND c = t + noise. Does the verdict
                    degrade toward "unmeasurable"? (the reverse demonstration, E4)

GATE (PLAN_V3 §G2): if ANY disjoint TabReD dataset becomes measurable under a sane
de-time-leaked / sparse representation → Claim A re-scopes to "deployed representation"
(survives, honestly). If overlap stays ≈0 even after removing time proxies → a STRONGER
result ("unmeasurable even without the time leakage"). add_timeproxy degrading a
measurable set → direct proof the dichotomy is a representation functional.

Model-light (sklearn). Server (TabReD); --synth-only runs the E4 demo anywhere.

    python scripts/run_representation.py --tabred sberbank_housing homecredit_default ecom_offers \
        homesite_insurance weather --elec2 --insects
    python scripts/run_representation.py --synth-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression  # noqa: E402

from src.analysis.drift_measure import (  # noqa: E402
    _stack, covariate_shift_auc, disde_iw_degeneration, concept_within_overlap,
)

TIMEPROXY_CORR = 0.30   # |corr(feature, t)| above this = "time-leaking" feature


def _row(tag, Xe, ye, Xl, yl, task, seed):
    cov = covariate_shift_auc(Xe, Xl, seed=seed).get("auc")
    deg = disde_iw_degeneration(Xe, Xl, seed=seed)
    con = concept_within_overlap(Xe, ye, Xl, yl, task, seed=seed)
    meas = bool(con.get("measurable"))
    return {"rep": tag, "n_feat": int(Xe.shape[1]), "cov_auc": cov,
            "overlap_mass": deg.get("overlap_mass"), "ess_pct": deg.get("ess_pct"),
            "measurable": meas,
            "concept_gap": con.get("concept_gap_within_overlap") if meas else None,
            "n_overlap_min": (min(con.get("n_overlap_early", 0), con.get("n_overlap_late", 0))
                              if meas else 0)}


def _timeproxy_mask(X, t):
    """columns with |corr(col, t)| > TIMEPROXY_CORR (the engineered time leakage)."""
    t = np.asarray(t, float)
    leak = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        c = X[:, j].astype(float)
        m = ~np.isnan(c)
        if m.sum() < 10 or np.nanstd(c) == 0:
            continue
        r = np.corrcoef(c[m], t[m])[0, 1]
        leak[j] = abs(r) > TIMEPROXY_CORR if np.isfinite(r) else False
    return leak


def _topk_mi(X, y, task, k):
    Xf = np.nan_to_num(X.astype(float))
    mi = (mutual_info_regression(Xf, y, random_state=0) if task == "regression"
          else mutual_info_classif(Xf, y, random_state=0))
    return np.argsort(-mi)[:min(k, X.shape[1])]


def reps_for_dataset(name, Xe, ye, Xl, yl, t_e, t_l, task, seed, ks=(5, 10, 20, 50)):
    X = np.concatenate([Xe, Xl]); t = np.concatenate([t_e, t_l]); y = np.concatenate([ye, yl])
    ne = len(ye)
    rows = [_row("full", Xe, ye, Xl, yl, task, seed)]
    # drop time-proxy features
    leak = _timeproxy_mask(X, t)
    if leak.any() and (~leak).any():
        keep = ~leak
        rows.append({**_row(f"drop_timeproxy(-{int(leak.sum())})",
                            Xe[:, keep], ye, Xl[:, keep], yl, task, seed)})
    # sparse top-k MI-with-y (computed on pooled; selection is y-driven not t-driven)
    for k in ks:
        if k >= X.shape[1]:
            continue
        sel = _topk_mi(X, y, task, k)
        rows.append({**_row(f"sparse_MI@{k}", Xe[:, sel], ye, Xl[:, sel], yl, task, seed)})
    return {"dataset": name, "task": task, "reps": rows}


# ---- E4 reverse demo (synthetic, runs anywhere): add a time-proxy to a measurable set ----
def e4_demo(seed=0, n=8000, d=6):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n))
    Xe_ = rng.normal(0, 1, (n // 2, d)); Xl_ = rng.normal(0, 1, (n - n // 2, d))
    te, tl = t[:n // 2], t[n // 2:]
    # real concept: rule flips early->late, P(x) identical -> measurable
    ye = (3 * Xe_[:, 0] + rng.normal(0, .4, len(Xe_)) > 0).astype(int)
    yl = (-3 * Xl_[:, 0] + rng.normal(0, .4, len(Xl_)) > 0).astype(int)
    base = _row("measurable_base", Xe_, ye, Xl_, yl, "binclass", seed)
    # append a time-proxy c = t + noise  -> should push toward "unmeasurable"
    ce = (te + rng.normal(0, .05, len(te)))[:, None]; cl = (tl + rng.normal(0, .05, len(tl)))[:, None]
    addc = _row("+timeproxy", np.hstack([Xe_, ce]), ye, np.hstack([Xl_, cl]), yl, "binclass", seed)
    return [base, addc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--tabred", nargs="*", default=[])
    ap.add_argument("--elec2", action="store_true")
    ap.add_argument("--insects", action="store_true")
    ap.add_argument("--insects-variant", default="incremental_balanced")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synth-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path("results/phase1/representation"); out_dir.mkdir(parents=True, exist_ok=True)

    def show(rows):
        for r in rows:
            g = r["concept_gap"]; gt = f"{g:+.3f}" if isinstance(g, (int, float)) else "   -"
            print(f"    {r['rep']:22s} nF={r['n_feat']:>4d} cov_AUC={r['cov_auc']:.3f} "
                  f"ovlap={r.get('overlap_mass', float('nan')):.3f} "
                  f"meas={str(r['measurable']):>5s} gap={gt} n_ov={r['n_overlap_min']}")

    results = []
    print("\n==== E4 reverse demo (add a time-proxy to a measurable set) ====")
    demo = e4_demo(args.seed); show(demo)
    print("  EXPECT: measurable_base measurable w/ large gap; +timeproxy -> overlap drops,")
    print("  verdict degrades toward unmeasurable = the dichotomy is a representation functional.")
    results.append({"dataset": "E4_demo", "reps": demo})
    if args.synth_only:
        (out_dir / "summary.json").write_text(json.dumps({"results": results}, indent=2, default=float))
        print(f"\n  wrote {out_dir}/summary.json"); return

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config); root = Path(cfg.data.root)
    jobs = []
    for ds in args.tabred:
        from src.data.tabred_loader import load_tabred
        jobs.append((ds, load_tabred(ds, root, split=cfg.experiment.split)))
    if args.elec2:
        from src.data.elec2_loader import load_elec2
        jobs.append(("elec2", load_elec2(split="temporal", seed=0)))
    if args.insects:
        from src.data.insects_loader import load_insects
        jobs.append((f"insects_{args.insects_variant}",
                     load_insects(variant=args.insects_variant, split="temporal", seed=0)))

    print("\n==== representation sweep (does de-time-leaking restore overlap?) ====")
    for name, data in jobs:
        t = data.train.t; med = float(np.median(t))
        X = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
        em, lm = t <= med, t > med
        print(f"\n  [{name}] task={data.task}")
        r = reps_for_dataset(name, X[em], data.train.y[em], X[lm], data.train.y[lm],
                             t[em], t[lm], data.task, args.seed)
        show(r["reps"]); results.append(r)
    print("\n  GATE: a disjoint TabReD set becoming measurable under drop_timeproxy/sparse")
    print("  => Claim A re-scopes to 'deployed representation' (survives). Staying overlap~0")
    print("  => stronger ('unmeasurable even de-time-leaked'). Either is a result; report it.")
    (out_dir / "summary.json").write_text(json.dumps({"results": results}, indent=2, default=float))
    print(f"\n  wrote {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
