"""V3.4 P1 — decompose Elec2's within-overlap concept gap (+0.146) into
concept vs serial-autocorrelation vs Bayes-noise drift (external red-team 반론 B/C).

The red-team's most dangerous point about Claim A: the headline within-overlap gap on
Elec2 (+0.146) may not be a P(y|x) RULE change but (C) the late-trained model exploiting
short-range serial autocorrelation that the early-trained model lacks (the Žliobaitė 2013
critique — applied here to the GAP, not the arm AUCs which §8/C2 already anchored), and/or
(B) a drift in conditional-entropy / Bayes noise (fixed rule, early-noisy → late-clean),
which the permutation placebo does NOT control. If the gap collapses below the noise floor
once autocorrelation/noise are removed, Elec2 must drop out of the concept anchors and
Claim A's positive evidence shrinks to INSECTS alone.

Three model-light probes (sklearn HGB; reuse the ESS-gated within-overlap measure):

  (1) THINNING sweep [C]: keep every s-th sample in stream order (stride s∈{1,5,25}).
      Adjacent kept samples are s steps apart, so SHORT-RANGE serial autocorrelation is
      destroyed while the LONG-RANGE early→late rule change is preserved. If the gap is
      autocorrelation-driven it shrinks with s; if it is a genuine rule change it survives.
  (2) LAGGED-LABEL ablation [C]: append y_{t-1} (predecessor label in stream order) as a
      feature to BOTH arms. If the gap is serial structure, the lag feature lets both arms
      capture it and the early-vs-late gap shrinks; if it is concept, the gap persists.
      Also report how predictive y_{t-1} is of y (the autocorrelation strength).
  (3) BAYES-NOISE proxy [B]: best-achievable accuracy (HGB 5-fold CV) on the early vs late
      halves. A large early→late jump in achievable accuracy = conditional-entropy/noise
      drift; we report (gap − achievable-accuracy-drift) as a rough concept-net-of-noise.
      (Proxy, clearly labelled — not a formal Bayes-error estimate.)

Decision (pre-registered): Elec2 stays a concept anchor iff the gap remains > the
gap_hygiene noise floor (0.034) AND its CI excludes that floor under thinning AND after the
lagged-label ablation. Otherwise we report Elec2 as autocorrelation/noise-confounded and
re-scope Claim A's positive evidence (W1).

    python scripts/run_elec2_decompose.py --elec2
    python scripts/run_elec2_decompose.py --elec2 --insects   # INSECTS as a designed-drift contrast
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.analysis.drift_measure import _stack, concept_within_overlap  # noqa: E402

NOISE_FLOOR = 0.034     # gap_hygiene/G1 noise-drift floor
TIMEPROXY_CORR = 0.30   # |corr(feature, t)| above this = engineered time-leak (drop it)


def _timeproxy_mask(X, t):
    t = np.asarray(t, float); leak = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        c = X[:, j].astype(float); m = ~np.isnan(c)
        if m.sum() < 10 or np.nanstd(c) == 0:
            continue
        r = np.corrcoef(c[m], t[m])[0, 1]
        leak[j] = abs(r) > TIMEPROXY_CORR if np.isfinite(r) else False
    return leak


def _ci95(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return [float(a[0]), float(a[0])] if len(a) else [float("nan"), float("nan")]
    m, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
    return [float(m - 1.96 * se), float(m + 1.96 * se)]


def _gap_seeds(Xe, ye, Xl, yl, task, seeds):
    """Returns (gaps over measurable seeds, mean ess_pct, abstain note if any)."""
    gs, esss, note = [], [], None
    for s in seeds:
        r = concept_within_overlap(Xe, ye, Xl, yl, task, seed=s)
        if r.get("ess_pct") is not None:
            esss.append(r["ess_pct"])
        if r.get("measurable"):
            gs.append(r["concept_gap_within_overlap"])
        elif note is None:
            note = r.get("note")
    return gs, (float(np.mean(esss)) if esss else None), note


def _verdict(gaps):
    if not gaps:
        return None, None, "abstain"
    ci = _ci95(gaps)
    if ci[0] > NOISE_FLOOR:
        v = "concept(>floor)"
    elif ci[1] < NOISE_FLOOR:
        v = "below-floor"
    else:
        v = "straddles-floor"
    return float(np.mean(gaps)), ci, v


def decompose(name, X, y, t, t_raw, task, seeds, strides=(1, 5, 25), de_time_leak=True):
    order = np.argsort(t_raw, kind="stable")          # stream order
    Xs, ys, ts = X[order], np.asarray(y)[order], np.asarray(t)[order]
    out = {"dataset": name, "task": task, "n": int(len(ys)), "de_time_leak": de_time_leak}
    # Elec2's FULL representation is un-checkable under the ess gate (time-proxy features
    # collapse overlap); concept lives on the DE-TIME-LEAKED rep (representation §6). So we
    # test the autocorrelation/noise question THERE, on the representation where the +0.074
    # concept actually exists, not on the un-checkable full rep.
    if de_time_leak:
        leak = _timeproxy_mask(Xs, ts)
        out["n_timeproxy_dropped"] = int(leak.sum()); out["n_feat_kept"] = int((~leak).sum())
        if leak.any() and (~leak).any():
            Xs = Xs[:, ~leak]

    # (1) THINNING sweep --------------------------------------------------------
    thin = []
    for s in strides:
        idx = np.arange(0, len(ys), s)
        Xk, yk = Xs[idx], ys[idx]
        nk = len(yk); half = nk // 2
        em = np.zeros(nk, bool); em[:half] = True; lm = ~em       # early / late by stream
        g, ess, note = _gap_seeds(Xk[em], yk[em], Xk[lm], yk[lm], task, seeds)
        mean, ci, v = _verdict(g)
        thin.append({"stride": s, "n_kept": nk, "gap_mean": mean, "gap_ci": ci,
                     "n_meas": len(g), "ess_pct": ess, "abstain_note": note, "verdict": v})
    out["thinning"] = thin

    # (2) LAGGED-LABEL ablation -------------------------------------------------
    # only meaningful for binclass (predecessor label as a numeric feature)
    lag = None
    if task == "binclass":
        ylag = np.concatenate([[ys[0]], ys[:-1]]).astype(float)   # predecessor in stream
        autocorr_auc = float(roc_auc_score(ys, ylag))             # how predictive is y_{t-1}?
        autocorr_auc = max(autocorr_auc, 1 - autocorr_auc)
        Xlag = np.hstack([Xs, ylag[:, None]])
        half = len(ys) // 2
        em = np.zeros(len(ys), bool); em[:half] = True; lm = ~em
        g_no, _, _ = _gap_seeds(Xs[em], ys[em], Xs[lm], ys[lm], task, seeds)      # baseline
        g_lag, _, _ = _gap_seeds(Xlag[em], ys[em], Xlag[lm], ys[lm], task, seeds)  # +lag feature
        m_no, ci_no, v_no = _verdict(g_no)
        m_lag, ci_lag, v_lag = _verdict(g_lag)
        lag = {"autocorr_auc_y_from_ylag": autocorr_auc,
               "gap_no_lag": m_no, "gap_no_lag_ci": ci_no,
               "gap_with_lag": m_lag, "gap_with_lag_ci": ci_lag,
               "shrink": (None if (m_no is None or m_lag is None) else float(m_no - m_lag)),
               "verdict_with_lag": v_lag}
    out["lagged_label"] = lag

    # (3) BAYES-NOISE proxy: achievable-accuracy drift early vs late ------------
    half = len(ys) // 2
    Xe, ye2 = Xs[:half], ys[:half]; Xl, yl2 = Xs[half:], ys[half:]
    def cv_acc(Xa, ya):
        Xa = np.nan_to_num(np.asarray(Xa, float))
        if task == "binclass" and len(np.unique(ya)) < 2:
            return float("nan")
        m = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        return float(np.mean(cross_val_score(m, Xa, ya, cv=3, scoring="accuracy")))
    acc_e, acc_l = cv_acc(Xe, ye2), cv_acc(Xl, yl2)
    base_mean = out["thinning"][0]["gap_mean"]
    out["bayes_noise_proxy"] = {
        "achievable_acc_early": acc_e, "achievable_acc_late": acc_l,
        "achievable_acc_drift": (float(acc_l - acc_e) if np.isfinite(acc_e) and np.isfinite(acc_l) else None),
        "note": "rough conditional-entropy proxy; a large late−early jump means part of the "
                "gap could be Bayes-noise reduction, not a rule change",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--elec2", action="store_true")
    ap.add_argument("--insects", action="store_true")
    ap.add_argument("--insects-variant", default="incremental_balanced")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seeds = [args.seed + i for i in range(max(1, args.n_seeds))]
    out_dir = Path("results/phase1/elec2_decompose"); out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.elec2:
        from src.data.elec2_loader import load_elec2
        jobs.append(("elec2", load_elec2(split="temporal", seed=0)))
    if args.insects:
        from src.data.insects_loader import load_insects
        jobs.append((f"insects_{args.insects_variant}",
                     load_insects(variant=args.insects_variant, split="temporal", seed=0)))
    if not jobs:
        print("nothing to do; pass --elec2 and/or --insects"); return

    results = []
    print(f"\n==== Elec2 gap decomposition (concept vs autocorr vs noise), {len(seeds)} seeds ====")
    def f3(x):                       # None-safe %+.3f
        return f"{x:+.3f}" if isinstance(x, (int, float)) else "  abstain"

    for name, data in jobs:
        X = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
        r = decompose(name, X, data.train.y, data.train.t, data.train.t_raw, data.task, seeds)
        results.append(r)
        print(f"\n  [{name}] task={r['task']}  (de-time-leaked: dropped "
              f"{r.get('n_timeproxy_dropped','?')} time-proxy feats, kept {r.get('n_feat_kept','?')})")
        print("   (1) THINNING (short-range autocorrelation removed as stride grows):")
        for c in r["thinning"]:
            gt = (f"{c['gap_mean']:+.3f}[{c['gap_ci'][0]:+.3f},{c['gap_ci'][1]:+.3f}]"
                  if c["gap_mean"] is not None else f"abstain({c.get('abstain_note','')[:24]})")
            es = c.get("ess_pct"); est = f"ess%={es:.1f}" if isinstance(es, (int, float)) else ""
            print(f"     stride={c['stride']:>2d} n={c['n_kept']:>6d}  {est:>10s}  gap={gt}  [{c['verdict']}]")
        lg = r["lagged_label"]
        if lg:
            print(f"   (2) LAGGED-LABEL: y_t-1→y AUC={lg['autocorr_auc_y_from_ylag']:.3f} "
                  f"(autocorr strength); gap no-lag {f3(lg['gap_no_lag'])} -> with-lag "
                  f"{f3(lg['gap_with_lag'])} (shrink {f3(lg['shrink'])}) [{lg['verdict_with_lag']}]")
        bn = r["bayes_noise_proxy"]
        print(f"   (3) BAYES-NOISE proxy: achievable acc early {f3(bn['achievable_acc_early'])} "
              f"-> late {f3(bn['achievable_acc_late'])} (drift {f3(bn['achievable_acc_drift'])})")
    print("\n  PRE-REGISTERED READ: Elec2 stays a concept anchor iff the gap stays > 0.034 with")
    print("  CI excluding the floor under thinning AND after the lagged-label ablation. If it")
    print("  collapses, Elec2 is autocorrelation/noise-confounded -> Claim A re-scopes (W1).")
    (out_dir / "summary.json").write_text(json.dumps({"floor": NOISE_FLOOR, "results": results},
                                                     indent=2, default=float))
    print(f"\n  wrote {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
