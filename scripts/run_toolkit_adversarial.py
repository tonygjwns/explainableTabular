"""V3.0 G3 — adversarial validation of the toolkit (earn "validated abstention"). PLAN_V3 §G3

The §14 4×4 PASS is circular: run_toolkit_validation.make() plants concept ONLY as a
global linear rotation (symmetric noise, balanced prior, full support, orthogonal
covariate) — the one family the AUC-gap estimator is built to detect (red-team D3).
This script runs the DGPs the validator never exercises and reports where the toolkit
gives the WRONG answer, so the paper can scope its abstention claim honestly:

  A1 subregion rule-flip   : real concept in a minority subregion. Global AUC averages
                             it out -> expect FALSE NEGATIVE (gap~0 though concept real).
  A2 noise-drift           : fixed boundary, early high / late low Bayes noise, P(x) same.
                             concept=0 -> expect mild FALSE POSITIVE (conditional-entropy leak).
  A2b covariate-correlated : covariate shift ENTANGLED with the rule axis (not orthogonal).
                             concept=0 -> does the gap fire falsely?
  A3 trajectory drift      : disjoint support, rule = sign(x1 - beta*t) drifts smoothly.
                             toolkit ABSTAINS (overlap~0); a time-aware (drift-prior) model
                             EXPLOITS it -> "unmeasurable != unexploitable".
  REP rotation             : plant concept, then rotate coords mixing rule into covariate
                             dims -> does the verdict change while concept is constant?

Each cell prints: toolkit verdict (measurable? gap) vs GROUND TRUTH, and a CLASSIFICATION
{PASS, FALSE_NEG, FALSE_POS, BLIND_SPOT}. Model-light (sklearn); runs anywhere.

    python scripts/run_toolkit_adversarial.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.analysis.drift_measure import (  # noqa: E402
    covariate_shift_auc, disde_iw_degeneration, concept_within_overlap,
)


def _toolkit(Xe, ye, Xl, yl, seed):
    cov = covariate_shift_auc(Xe, Xl, seed=seed).get("auc")
    deg = disde_iw_degeneration(Xe, Xl, seed=seed)
    con = concept_within_overlap(Xe, ye, Xl, yl, "binclass", seed=seed)
    meas = bool(con.get("measurable"))
    return {"cov_auc": cov, "overlap_mass": deg.get("overlap_mass"), "measurable": meas,
            "gap": con.get("concept_gap_within_overlap") if meas else None}


# ---------- DGPs ----------
def a1_subregion(seed, n=8000, d=6, frac=0.16):
    """concept flips only in subregion x5>=q (frac mass); P(x) identical. GT: concept REAL."""
    rng = np.random.default_rng(seed)
    Xe = rng.normal(0, 1, (n, d)); Xl = rng.normal(0, 1, (n, d))
    q = np.quantile(Xl[:, 5], 1 - frac)
    ye = (4 * Xe[:, 0] + rng.normal(0, .3, n) > 0).astype(int)
    flip = Xl[:, 5] >= q
    rule = np.where(flip, -4 * Xl[:, 0], 4 * Xl[:, 0])
    yl = (rule + rng.normal(0, .3, n) > 0).astype(int)
    return Xe, ye, Xl, yl, "concept REAL (subregion)"


def a2_noise(seed, n=8000, d=6):
    """fixed boundary, early noisy / late clean labels, P(x) same. GT: concept=0."""
    rng = np.random.default_rng(seed)
    Xe = rng.normal(0, 1, (n, d)); Xl = rng.normal(0, 1, (n, d))
    ye = (1.0 * Xe[:, 0] + rng.normal(0, 1, n) > 0).astype(int)
    yl = (6.0 * Xl[:, 0] + rng.normal(0, 1, n) > 0).astype(int)
    return Xe, ye, Xl, yl, "concept=0 (noise drift)"


def a2b_covcorr(seed, n=8000, d=6, mu=1.2):
    """covariate shift ALONG the rule axis (x0), boundary fixed. GT: concept=0."""
    rng = np.random.default_rng(seed)
    Xe = rng.normal(0, 1, (n, d)); Xl = rng.normal(0, 1, (n, d)); Xl[:, 0] += mu  # entangled
    ye = (2 * Xe[:, 0] + rng.normal(0, .4, n) > 0).astype(int)
    yl = (2 * Xl[:, 0] + rng.normal(0, .4, n) > 0).astype(int)   # SAME rule
    return Xe, ye, Xl, yl, "concept=0 (covariate||rule)"


def a3_trajectory(seed, n=9000, d=4, c=9.0):
    """disjoint support (mean drifts c*t); rule sign(x1 - beta*t) drifts smoothly.
    GT: concept EXPLOITABLE by a time-aware model via extrapolation. Returns extra (Xmid)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n))
    X = rng.normal(0, 1, (n, d)); X[:, 0] += c * t            # x0 carries time -> disjoint
    beta = 3.0
    y = (X[:, 1] - beta * t + rng.normal(0, .3, n) > 0).astype(int)
    early, mid, late = t < 1 / 3, (t >= 1 / 3) & (t < 2 / 3), t >= 2 / 3
    return (X, y, t, early, mid, late)


def a3_exploitability(seed):
    """toolkit verdict on (early,late) vs a time-aware model's late accuracy (trained early+mid)."""
    X, y, t, early, mid, late = a3_trajectory(seed)
    tk = _toolkit(X[early], y[early], X[late], y[late], seed)
    tr = early | mid
    Xtr = np.column_stack([X[tr], t[tr]])                      # time-AWARE features
    Xte = np.column_stack([X[late], t[late]])
    Xtr0 = X[tr]; Xte0 = X[late]                               # time-UNAWARE features
    def auc(Xa, Xb):
        m = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xa, y[tr])
        return roc_auc_score(y[late], m.predict_proba(Xb)[:, 1])
    aware, unaware = auc(Xtr, Xte), auc(Xtr0, Xte0)
    return tk, aware, unaware


def rep_rotation(seed, n=6000, d=6, theta=70.0, mu=0.0):
    """plant concept (rotation), then rotate coords mixing rule(0,1) into dims (2,3).
    GT: concept constant; verdict should ideally be invariant."""
    rng = np.random.default_rng(seed)
    Xe = rng.normal(0, 1, (n, d)); Xl = rng.normal(0, 1, (n, d))
    th = np.radians(theta); w0 = np.zeros(d); w0[0] = 1
    wl = np.zeros(d); wl[0] = np.cos(th); wl[1] = np.sin(th)
    ye = (2 * (Xe @ w0) + rng.normal(0, .4, n) > 0).astype(int)
    yl = (2 * (Xl @ wl) + rng.normal(0, .4, n) > 0).astype(int)
    # rotate coordinates: mix dims (0,1) into (2,3) by 45deg (invertible, label-preserving)
    R = np.eye(d); a = np.radians(45)
    R[0, 0] = R[2, 2] = np.cos(a); R[0, 2] = -np.sin(a); R[2, 0] = np.sin(a)
    R[1, 1] = R[3, 3] = np.cos(a); R[1, 3] = -np.sin(a); R[3, 1] = np.sin(a)
    return Xe, ye, Xl, yl, Xe @ R.T, Xl @ R.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()
    out_dir = Path("results/phase1/toolkit_adversarial"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def avg_toolkit(gen):
        gaps, meas, covs = [], [], []
        for s in range(args.n_seeds):
            Xe, ye, Xl, yl, gt = gen(s)
            tk = _toolkit(Xe, ye, Xl, yl, s)
            meas.append(tk["measurable"]); covs.append(tk["cov_auc"])
            gaps.append(tk["gap"] if tk["gap"] is not None else np.nan)
        return float(np.nanmean(gaps)), float(np.mean(meas)), float(np.mean(covs)), gt

    print("\n==== toolkit ADVERSARIAL cells (where does it give the WRONG answer?) ====")
    print(f"  {'cell':18s}{'GT':28s}{'toolkit gap':>12s}{'meas%':>7s}  classification")

    # A1 false negative
    g, m, cv, gt = avg_toolkit(a1_subregion)
    cls = "FALSE_NEG" if (m > 0.5 and abs(g) < 0.03) else "ok"
    print(f"  {'A1 subregion':18s}{gt:28s}{g:+12.4f}{100*m:7.0f}  {cls} (real concept read as ~0)")
    rows.append({"cell": "A1_subregion", "gt": gt, "gap": g, "meas_frac": m, "class": cls})

    # A2 / A2b false positive
    for nm, gen in [("A2 noise-drift", a2_noise), ("A2b cov||rule", a2b_covcorr)]:
        g, m, cv, gt = avg_toolkit(gen)
        cls = "FALSE_POS" if (m > 0.5 and abs(g) > 0.03) else "ok"
        print(f"  {nm:18s}{gt:28s}{g:+12.4f}{100*m:7.0f}  {cls} (concept=0; gap should be ~0)")
        rows.append({"cell": nm, "gt": gt, "gap": g, "meas_frac": m, "class": cls})

    # A3 unmeasurable != unexploitable
    tks, aw, un = [], [], []
    for s in range(args.n_seeds):
        tk, a, u = a3_exploitability(s)
        tks.append(tk["measurable"]); aw.append(a); un.append(u)
    meas = float(np.mean(tks)); aware, unaware = float(np.mean(aw)), float(np.mean(un))
    blind = (meas < 0.5) and (aware - unaware > 0.05)
    print(f"\n  A3 trajectory: toolkit measurable%={100*meas:.0f} (abstains); "
          f"time-AWARE late-AUC={aware:.3f} vs time-UNAWARE={unaware:.3f} (gain {aware-unaware:+.3f})")
    print(f"     -> {'BLIND_SPOT: unmeasurable but EXPLOITABLE by a time-aware model' if blind else 'ok'}")
    rows.append({"cell": "A3_trajectory", "toolkit_measurable_frac": meas,
                 "aware_late_auc": aware, "unaware_late_auc": unaware,
                 "exploitable_gain": aware - unaware, "class": "BLIND_SPOT" if blind else "ok"})

    # REP rotation invariance
    base_g, rot_g, base_m, rot_m = [], [], [], []
    for s in range(args.n_seeds):
        Xe, ye, Xl, yl, XeR, XlR = rep_rotation(s)
        b = _toolkit(Xe, ye, Xl, yl, s); r = _toolkit(XeR, ye, XlR, yl, s)
        base_g.append(b["gap"] if b["gap"] is not None else np.nan); base_m.append(b["measurable"])
        rot_g.append(r["gap"] if r["gap"] is not None else np.nan); rot_m.append(r["measurable"])
    bg, rg = float(np.nanmean(base_g)), float(np.nanmean(rot_g))
    print(f"\n  REP rotation: gap base={bg:+.3f} (meas%={100*np.mean(base_m):.0f}) vs "
          f"rotated={rg:+.3f} (meas%={100*np.mean(rot_m):.0f})  "
          f"-> {'representation-SENSITIVE' if abs(bg-rg) > 0.03 else 'invariant-ish'}")
    rows.append({"cell": "REP_rotation", "gap_base": bg, "gap_rotated": rg,
                 "class": "REP_SENSITIVE" if abs(bg - rg) > 0.03 else "ok"})

    print("\n  READ: any FALSE_NEG/FALSE_POS/BLIND_SPOT must be reported as a scope limit of")
    print("  the toolkit; the abstention claim holds only outside these modes.")
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2, default=float))
    print(f"\n  wrote {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
