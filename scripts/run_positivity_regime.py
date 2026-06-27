"""V4 pre-test — is temporal tabular shift a HARMFUL positivity-failure regime, or a
benign preprocessing/feature-engineering artifact? (the pivot's make-or-break)

The robust surviving fact: on the deployed (TabReD-standard) representation, early/late
feature supports become nearly disjoint (cov-AUC≈1, overlap→0). The candidate new spine:
"real tabular temporal shift is an out-of-support / positivity-failure regime where the
standard shift-correction toolkit (importance weighting, conformal) silently breaks."

But the user's sharp objection must be ruled out FIRST: maybe the disjointness is a
PREPROCESSING ARTIFACT (a clock/ID feature leaking time), or it lives only in nuisance
feature directions (BENIGN: predictive directions still overlap, downstream is fine). Only
if the disjointness reaches the LABEL-RELEVANT directions AND breaks downstream correction
is the regime HARMFUL — and the new spine real. This probe measures exactly that, per
dataset, model-light (sklearn). Three blocks:

  (1) BUG-CLEAN: how much of the separability is a clock/ID artifact?
      - n_clocklike = features with |corr(feat,t)| > 0.95 (index/timestamp leaks)
      - cov_auc_raw vs cov_auc_no_timeproxy (drop |corr(feat,t)|>0.3): if it collapses to
        ~0.5, the "shift" was just a clock; if it stays high, it is real multivariate shift.
  (2) BENIGN vs HARMFUL: is the disjointness in the predictive directions?
      - raw overlap (all features) vs label-relevant overlap (top-k MI-with-y features) vs
        predictive-coordinate overlap (1-D out-of-fold ŷ). HARMFUL := label-relevant cov-AUC
        still high (the directions that matter are disjoint); BENIGN := they overlap.
  (3) DOWNSTREAM BREAKAGE: do standard corrections actually fail where it is harmful?
      - IW degeneration (early→late density-ratio ESS%, single-point leverage)
      - split-conformal: calibrate at nominal 90% on held-out EARLY, deploy on LATE, measure
        actual coverage. Under-coverage = conformal broke under the shift.

CROSS-DATASET pre-registered KILL: if label-relevant disjointness does NOT predict downstream
breakage (Spearman(labelrel_cov_auc, conformal under-coverage) n.s.) AND conformal coverage
stays ~nominal everywhere -> there is no HARM axis -> the positivity-failure spine is killed
(report the benign/artifact finding honestly and fall back to the measurement/datasheet paper).

    python scripts/run_positivity_regime.py --tabred sberbank_housing homecredit_default \
        ecom_offers homesite_insurance weather cooking_time maps_routing delivery_eta --elec2 --insects
    python scripts/run_positivity_regime.py --synth-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression  # noqa: E402
from sklearn.model_selection import cross_val_predict, train_test_split  # noqa: E402

from src.analysis.drift_measure import _stack, covariate_shift_auc, disde_iw_degeneration  # noqa: E402

TIMEPROXY_CORR = 0.30
CLOCK_CORR = 0.95


def _corr_t(X, t):
    t = np.asarray(t, float); out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        c = X[:, j].astype(float); m = ~np.isnan(c)
        if m.sum() < 10 or np.nanstd(c) == 0:
            continue
        r = np.corrcoef(c[m], t[m])[0, 1]
        out[j] = abs(r) if np.isfinite(r) else 0.0
    return out


def _topk_mi(X, y, task, k, seed=0):
    Xf = np.nan_to_num(np.asarray(X, float))
    mi = (mutual_info_regression(Xf, y, random_state=seed) if task == "regression"
          else mutual_info_classif(Xf, y, random_state=seed))
    return np.argsort(-mi)[:min(k, X.shape[1])]


def _pred_coord(X, y, task, seed):
    """Out-of-fold predictive coordinate ŷ (1-D): the model's view of each row."""
    Xf = np.nan_to_num(np.asarray(X, float))
    if task == "regression":
        m = HistGradientBoostingRegressor(max_iter=200, random_state=seed)
        return cross_val_predict(m, Xf, y, cv=4).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return None
    m = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    p = cross_val_predict(m, Xf, y, cv=4, method="predict_proba")
    return p[:, 1:2] if p.shape[1] == 2 else p          # binclass: P(1); multi: full proba


def _transfer_perf_drop(Xe, ye, Xl, yl, task, seed):
    """Early-trained model: score on held-out EARLY vs on LATE. Oriented drop (>0 = late
    worse) = the DIRECT harm of the shift (does disjointness actually degrade prediction?)."""
    from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
    Xe = np.nan_to_num(np.asarray(Xe, float)); Xl = np.nan_to_num(np.asarray(Xl, float))
    Xtr, Xho, ytr, yho = train_test_split(Xe, ye, test_size=0.4, random_state=seed)
    if task == "regression":
        m = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(Xtr, ytr)
        rmse = lambda X, y: float(np.sqrt(mean_squared_error(y, m.predict(X))))
        return float(rmse(Xl, yl) - rmse(Xho, yho))           # >0 = late worse
    if len(np.unique(ytr)) < 2:
        return None
    m = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(Xtr, ytr)
    if task == "binclass":
        sc = lambda X, y: roc_auc_score(y, m.predict_proba(X)[:, 1]) if len(np.unique(y)) > 1 else float("nan")
    else:
        sc = lambda X, y: accuracy_score(y, m.predict(X))
    s_ho, s_l = sc(Xho, yho), sc(Xl, yl)
    return float(s_ho - s_l) if np.isfinite(s_ho) and np.isfinite(s_l) else None   # >0 = late worse


def _conformal_late_coverage(Xe, ye, Xl, yl, task, seed, alpha=0.10):
    """Split-conformal calibrated on held-out EARLY, deployed on LATE. Returns actual late
    coverage (nominal 1-alpha). Under-coverage => the correction broke under the shift."""
    Xe = np.nan_to_num(np.asarray(Xe, float)); Xl = np.nan_to_num(np.asarray(Xl, float))
    Xtr, Xcal, ytr, ycal = train_test_split(Xe, ye, test_size=0.4, random_state=seed)
    if task == "regression":
        m = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(Xtr, ytr)
        scal = np.abs(ycal - m.predict(Xcal))                       # nonconformity = |resid|
        q = np.quantile(scal, 1 - alpha)
        lo, hi = m.predict(Xl) - q, m.predict(Xl) + q
        return float(np.mean((yl >= lo) & (yl <= hi)))
    if len(np.unique(ytr)) < 2:
        return None
    m = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(Xtr, ytr)
    cls = list(m.classes_); idx = {c: i for i, c in enumerate(cls)}
    pcal = m.predict_proba(Xcal)
    scal = 1.0 - pcal[np.arange(len(ycal)), [idx.get(c, 0) for c in ycal]]   # 1 - p(true)
    q = np.quantile(scal, 1 - alpha)
    pl = m.predict_proba(Xl)
    inset = pl >= (1 - q)                                            # prediction set per row
    cover = [inset[i, idx[c]] if c in idx else False for i, c in enumerate(yl)]
    return float(np.mean(cover))


def assess(name, Xe, ye, Xl, yl, task, seed=0):
    ne, nl = len(ye), len(yl)
    t_all = np.concatenate([np.zeros(ne), np.ones(nl)])     # proxy time = early/late label
    X_all = np.concatenate([Xe, Xl]); y_all = np.concatenate([ye, yl])
    # (1) bug-clean
    corr = _corr_t(X_all, t_all)
    n_clock = int((corr > CLOCK_CORR).sum()); leak = corr > TIMEPROXY_CORR
    cov_raw = covariate_shift_auc(Xe, Xl, seed=seed).get("auc")
    keep = ~leak
    cov_notp = (covariate_shift_auc(Xe[:, keep], Xl[:, keep], seed=seed).get("auc")
                if keep.sum() >= 2 and keep.any() else None)
    # (2) benign vs harmful: label-relevant + predictive-coordinate overlap
    sel = _topk_mi(X_all, y_all, task, k=10, seed=seed)
    cov_lr = covariate_shift_auc(Xe[:, sel], Xl[:, sel], seed=seed).get("auc")
    deg_lr = disde_iw_degeneration(Xe[:, sel], Xl[:, sel], seed=seed)
    pc = _pred_coord(X_all, y_all, task, seed)
    cov_pc = (covariate_shift_auc(pc[:ne], pc[ne:], seed=seed).get("auc") if pc is not None else None)
    # (3) downstream breakage — the DIRECT harm: does the disjointness degrade prediction?
    deg_raw = disde_iw_degeneration(Xe, Xl, seed=seed)
    cov_full = _conformal_late_coverage(Xe, ye, Xl, yl, task, seed)
    cov_lr_cf = _conformal_late_coverage(Xe[:, sel], ye, Xl[:, sel], yl, task, seed)
    perf_drop = _transfer_perf_drop(Xe, ye, Xl, yl, task, seed)
    under = (0.90 - cov_full) if isinstance(cov_full, (int, float)) else None
    # HARMFUL := the shift actually hurts (conformal under-covers >5pt OR perf drops materially).
    # BENIGN := disjoint on paper but prediction/coverage hold (a nuisance/artifact disjointness).
    harmful = bool((under is not None and under > 0.05)
                   or (perf_drop is not None and perf_drop > (0.5 if task == "regression" else 0.05)))
    return {"dataset": name, "task": task, "n_feat": int(Xe.shape[1]), "n_early": ne, "n_late": nl,
            "n_clocklike": n_clock, "n_timeproxy": int(leak.sum()),
            "cov_auc_raw": cov_raw, "cov_auc_no_timeproxy": cov_notp,
            "cov_auc_labelrel": cov_lr, "cov_auc_predcoord": cov_pc,
            "overlap_raw": deg_raw.get("overlap_mass"), "ess_raw": deg_raw.get("ess_pct"),
            "overlap_labelrel": deg_lr.get("overlap_mass"), "ess_labelrel": deg_lr.get("ess_pct"),
            "conformal_cov_full": cov_full, "conformal_cov_labelrel": cov_lr_cf,
            "perf_drop_late_vs_early": perf_drop, "harmful": harmful}


def _split(data):
    t = data.train.t; med = float(np.median(t))
    X = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
    em, lm = t <= med, t > med
    return X[em], data.train.y[em], X[lm], data.train.y[lm], data.task


# ---- synthetic smoke: BENIGN (nuisance shift) vs HARMFUL (predictive-dir shift) ----
def _synth(kind, seed=0, n=8000, d=20):
    """d>>k so label-relevant != full. Nonlinear rule on 3 predictive dims so that, under
    HARMFUL shift (predictive dims move out of support), the early model EXTRAPOLATES WRONG
    -> conformal under-covers. BENIGN moves only nuisance dims -> model & coverage unaffected."""
    rng = np.random.default_rng(seed)
    pred = [0, 1, 2]; nuis = list(range(3, d))
    Xe = rng.normal(0, 1, (n, d)); Xl = rng.normal(0, 1, (n, d))
    if kind == "benign":
        Xl[:, nuis] += 6.0                          # only nuisance dims shift
    else:
        Xl[:, pred] = rng.normal(6.0, 1.0, (n, len(pred)))   # PREDICTIVE dims shift out of support

    def rule(X):                                    # nonlinear -> extrapolation is wrong off-support
        z = np.sin(2.0 * X[:, 0]) + 0.8 * X[:, 1] - 0.8 * X[:, 2]
        return (z + rng.normal(0, .3, len(X)) > 0).astype(int)
    return Xe, rule(Xe), Xl, rule(Xl), "binclass"


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
    out_dir = Path("results/phase1/positivity_regime"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def show(r):
        f = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "  -"
        print(f"  {r['dataset'][:20]:20s} clk={r['n_clocklike']:>2d} "
              f"covRAW={f(r['cov_auc_raw'])} noTP={f(r['cov_auc_no_timeproxy'])} "
              f"covLR={f(r['cov_auc_labelrel'])} | conf={f(r['conformal_cov_full'])} "
              f"drop={f(r['perf_drop_late_vs_early'])} [{'HARMFUL' if r['harmful'] else 'benign'}]")

    if args.synth_only:
        print("\n==== SYNTH smoke: BENIGN (nuisance shift) vs HARMFUL (predictive-dir shift) ====")
        print("  EXPECT benign: covRAW high, covLR low, conformal ~0.90.  harmful: covLR high, conformal << 0.90")
        for kind in ("benign", "harmful"):
            Xe, ye, Xl, yl, task = _synth(kind, args.seed)
            r = assess(f"synth_{kind}", Xe, ye, Xl, yl, task, args.seed); rows.append(r); show(r)
        (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2, default=float))
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

    print("\n==== POSITIVITY REGIME probe (bug-clean / benign-vs-harmful / downstream breakage) ====")
    for name, data in jobs:
        Xe, ye, Xl, yl, task = _split(data)
        r = assess(name, Xe, ye, Xl, yl, task, args.seed); rows.append(r); show(r)

    # cross-dataset: does label-relevant disjointness predict conformal under-coverage?
    usable = [r for r in rows if r["cov_auc_raw"] is not None
              and (r["conformal_cov_full"] is not None or r["perf_drop_late_vs_early"] is not None)]
    print("\n  ==== cross-dataset: does raw disjointness predict HARM? ====")
    stat = {}
    if len(usable) >= 3:
        cov = np.array([r["cov_auc_raw"] for r in usable])
        drop = np.array([r["perf_drop_late_vs_early"] if r["perf_drop_late_vs_early"] is not None
                         else 0.0 for r in usable])
        rho, p = spearmanr(cov, drop)
        n_harm = sum(r["harmful"] for r in usable)
        n_disjoint = sum((r["cov_auc_raw"] or 0) > 0.9 for r in usable)
        n_disjoint_harm = sum(((r["cov_auc_raw"] or 0) > 0.9) and r["harmful"] for r in usable)
        stat = {"spearman_cov_auc_perfdrop": float(rho), "p": float(p), "n": len(usable),
                "n_harmful": int(n_harm), "n_disjoint_cov>0.9": int(n_disjoint),
                "n_disjoint_AND_harmful": int(n_disjoint_harm)}
        print(f"  Spearman(cov_auc_raw, perf_drop) = {rho:+.3f} (p={p:.3f}), n={len(usable)}")
        print(f"  disjoint (cov>0.9): {n_disjoint}/{len(usable)} | of those, HARMFUL: {n_disjoint_harm}")
        print(f"  total HARMFUL: {n_harm}/{len(usable)}")
        print("  PRE-REG KILL: if disjoint datasets are NOT harmful (n_disjoint_AND_harmful≈0 AND rho n.s.)")
        print("  => the 261-dim disjointness is a BENIGN nuisance/preprocessing artifact => positivity-spine")
        print("  killed, fall back to the measurement/datasheet framing. Else: HARMFUL regime real => spine viable.")
    else:
        print("  too few datasets for the correlation.")
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "cross": stat}, indent=2, default=float))
    print(f"\n  wrote {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
