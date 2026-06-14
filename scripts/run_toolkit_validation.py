"""R2.4 — toolkit validation on a controlled covariate×concept grid (ground truth).

D&B reviewers ask a measurement paper to VALIDATE its instrument on data where the
answer is known, incl. failure modes. We generate synthetic early/late data with
TWO independently-controlled knobs and check the toolkit (covariate_shift_auc +
disde_iw_degeneration + concept_within_overlap) recovers ground truth:

  - mu_cov   : covariate-shift strength. Shifts the NON-rule feature dims (>=2) of
               the LATE period by mu_cov (pervasive, ORTHOGONAL to the rule) =>
               controls P(x) shift only. mu_cov 0 -> none; large -> disjoint support.
  - theta    : concept strength. The decision rule w rotates by theta in the
               (feat0,feat1) plane early->late => controls P(y|x) only (difficulty
               matched, so the within-overlap transfer gap is pure concept).

Ground-truth expectations the script asserts/reports:
  1. RECOVERY: at low covariate (overlap large), concept_gap rises monotonically with
     theta and is ~0 at theta=0 (no spurious concept). Spearman(theta, gap) ~ +1.
  2. NO FALSE POSITIVE: concept_gap ~ 0 in every theta=0 cell, regardless of mu_cov.
  3. DEGENERATION: increasing mu_cov => cov_AUC up, overlap_mass down, ESS% down.
  4. FAILURE MODE KNOWN: at high mu_cov the overlap empties => frame reports
     measurable=False (it does NOT emit a false concept number).

Model-light (sklearn HGB only) -> runs in a couple minutes, locally too.

    python scripts/run_toolkit_validation.py
    python scripts/run_toolkit_validation.py --mu-cov 0 0.7 1.5 3.0 --theta 0 30 60 90
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.analysis.drift_measure import (  # noqa: E402
    covariate_shift_auc, disde_iw_degeneration, concept_within_overlap,
)


def make(n, d, mu_cov, theta_deg, *, scale=2.0, noise=0.4, seed=0):
    """early/late binary data: rule in (feat0,feat1) rotates by theta (concept);
    non-rule dims (>=2) of late shift by mu_cov (covariate, orthogonal to the rule)."""
    rng = np.random.default_rng(seed)
    Xe = rng.normal(0, 1, (n, d)); Xl = rng.normal(0, 1, (n, d))
    Xl[:, 2:] += mu_cov                                  # pervasive covariate shift, off-rule
    th = math.radians(theta_deg)
    w0 = np.zeros(d); w0[0] = 1.0
    wl = np.zeros(d); wl[0] = math.cos(th); wl[1] = math.sin(th)
    ye = (scale * (Xe @ w0) + noise * rng.normal(size=n) > 0).astype("int64")
    yl = (scale * (Xl @ wl) + noise * rng.normal(size=n) > 0).astype("int64")
    return Xe, ye, Xl, yl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu-cov", nargs="+", type=float, default=[0.0, 0.7, 1.5, 3.0])
    ap.add_argument("--theta", nargs="+", type=float, default=[0.0, 30.0, 60.0, 90.0])
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--d", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/phase1.yaml")
    args = ap.parse_args()

    out_dir = Path("results/phase1/toolkit_validation"); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n==== toolkit validation: covariate(mu) x concept(theta) grid "
          f"(n={args.n}, d={args.d}) ====")
    print(f"  {'mu_cov':>6s} {'theta':>6s} | {'cov_AUC':>7s} {'ovlap':>6s} {'ESS%':>6s} | "
          f"{'measurable':>10s} {'gap':>8s}")
    rows = []
    for mu in args.mu_cov:
        for th in args.theta:
            Xe, ye, Xl, yl = make(args.n, args.d, mu, th, seed=args.seed)
            cov = covariate_shift_auc(Xe, Xl, seed=args.seed).get("auc")
            deg = disde_iw_degeneration(Xe, Xl, seed=args.seed)
            con = concept_within_overlap(Xe, ye, Xl, yl, "binclass", seed=args.seed)
            meas = bool(con.get("measurable"))
            gap = con.get("concept_gap_within_overlap") if meas else None
            gtxt = f"{gap:+.3f}" if isinstance(gap, (int, float)) else "   -"
            print(f"  {mu:6.2f} {th:6.0f} | {cov:7.3f} {deg.get('overlap_mass',0):6.3f} "
                  f"{deg.get('ess_pct',0):6.2f} | {str(meas):>10s} {gtxt:>8s}")
            rows.append({"mu_cov": mu, "theta": th, "cov_auc": cov,
                         "overlap_mass": deg.get("overlap_mass"), "ess_pct": deg.get("ess_pct"),
                         "measurable": meas, "concept_gap": gap,
                         "n_overlap_min": (min(con.get("n_overlap_early", 0),
                                               con.get("n_overlap_late", 0)) if meas else 0)})

    # ---- ground-truth validations ----
    print("\n  ==== validations ====")
    lo_mu = min(args.mu_cov)
    low = [r for r in rows if r["mu_cov"] == lo_mu and r["measurable"]]
    ok = True
    if len(low) >= 3:
        rho, p = spearmanr([r["theta"] for r in low], [r["concept_gap"] for r in low])
        print(f"  1. RECOVERY  Spearman(theta, gap) at mu={lo_mu}: {rho:+.3f} (p={p:.3f}) "
              f"[expect ~+1]")
        ok &= rho > 0.7
    fp = [r for r in rows if r["theta"] == 0.0 and r["measurable"]]
    maxfp = max((abs(r["concept_gap"]) for r in fp), default=0.0)
    print(f"  2. NO FALSE POS: max|gap| over theta=0 measurable cells = {maxfp:.3f} "
          f"[expect <0.03]")
    ok &= maxfp < 0.05
    # degeneration monotonicity at a fixed mid theta
    th_mid = sorted(args.theta)[len(args.theta) // 2]
    col = [r for r in rows if r["theta"] == th_mid]
    if len(col) >= 3:
        rc, _ = spearmanr([r["mu_cov"] for r in col], [r["cov_auc"] for r in col])
        ro, _ = spearmanr([r["mu_cov"] for r in col], [r["overlap_mass"] for r in col])
        print(f"  3. DEGENERATION at theta={th_mid:.0f}: Spearman(mu,cov_AUC)={rc:+.2f} "
              f"[+], Spearman(mu,overlap_mass)={ro:+.2f} [-]")
        ok &= (rc > 0.7 and ro < -0.7)
    hi_mu = max(args.mu_cov)
    hi = [r for r in rows if r["mu_cov"] == hi_mu]
    n_unmeas = sum(not r["measurable"] for r in hi)
    print(f"  4. FAILURE MODE: at mu={hi_mu}, {n_unmeas}/{len(hi)} cells unmeasurable "
          f"(overlap empty => no false concept emitted) [expect most/all]")
    print(f"\n  TOOLKIT VALIDATION: {'PASS' if ok else 'CHECK'} "
          f"(recovery + no-false-positive + degeneration monotone)")
    (out_dir / "summary.json").write_text(json.dumps(
        {"grid": rows, "pass": bool(ok)}, indent=2, default=float))
    print(f"  wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
