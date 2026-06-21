"""V3.0 G1 — decisive controls for the within-overlap concept gap (PLAN_V3 §G1).

The red-team's single most damaging finding (RED_TEAM R1 + D3-A2): the transfer gap
`AUC_late − AUC_early` may measure a TRAIN/TEST HOME-FIELD advantage (the late-trained
model is tested on its own distribution) and/or label-noise/prior drift, NOT a P(y|x)
change. If so, the flagship positives (Elec2 +0.132, INSECTS +0.144) are confounded.

This script runs the controls that GATE the whole rebuild:
  (1) PLACEBO null: permute the early/late label within the overlap band (same region,
      no real early→late structure). A real concept gap must DROP to ~0 here; a residual
      = the home-field/N-asymmetry bias floor. Reports true gap vs placebo, multi-seed,
      with bootstrap CIs and the bias-corrected (true − placebo) effect.
  (2) SYNTHETIC label-noise-drift null (A2): fixed decision boundary, early high Bayes
      noise / late low noise, identical P(x). True concept = 0. If the gap fires large
      here, the estimator confounds conditional-entropy drift with concept.
  (3) SYNTHETIC prior-shift null: fixed P(y|x), P(y) changes. True concept = 0.

GATE (pre-registered, PLAN_V3 §G1): a real-data positive survives iff its CI lies above
the placebo null AND the synthetic nulls read ~0. Otherwise → PLAN_V3 §0.1 branch.

Model-light (sklearn HGB). Run on the server (TabReD data) or locally for the synthetics.

    python scripts/run_gap_controls.py --elec2 --insects --tabred cooking_time maps_routing
    python scripts/run_gap_controls.py --synth-only          # nulls only, runs anywhere
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.analysis.drift_measure import _stack, concept_within_overlap  # noqa: E402


def _gap(Xe, ye, Xl, yl, task, seed, permute):
    r = concept_within_overlap(Xe, ye, Xl, yl, task, seed=seed, permute_time=permute)
    return r.get("concept_gap_within_overlap") if r.get("measurable") else None


def real_controls(name, Xe, ye, Xl, yl, task, n_seeds):
    """true gap vs placebo (permuted-time) null, over n_seeds; bootstrap-style CIs."""
    true = np.array([g for s in range(n_seeds)
                     if (g := _gap(Xe, ye, Xl, yl, task, s, False)) is not None])
    plac = np.array([g for s in range(n_seeds)
                     if (g := _gap(Xe, ye, Xl, yl, task, s, True)) is not None])
    def ci(a):
        if len(a) < 2:
            return (float("nan"), float("nan"))
        m, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
        return (m - 1.96 * se, m + 1.96 * se)
    t_ci, p_ci = ci(true), ci(plac)
    corrected = float(true.mean() - plac.mean()) if len(true) and len(plac) else float("nan")
    # is the true gap above the placebo null? (CI separation)
    survives = bool(len(true) and len(plac) and t_ci[0] > p_ci[1])
    return {"dataset": name, "task": task, "n_seeds_used": int(len(true)),
            "true_gap_mean": float(true.mean()) if len(true) else None, "true_gap_ci": t_ci,
            "placebo_mean": float(plac.mean()) if len(plac) else None, "placebo_ci": p_ci,
            "bias_corrected_gap": corrected, "survives_placebo": survives}


# ---------- synthetic nulls (true concept = 0) ----------
def synth_noise_drift(n=8000, d=6, seed=0):
    """A2: fixed boundary y=sign(a*x0); early HIGH Bayes noise, late LOW. P(x) identical.
    True P(y|x) RULE unchanged (argmax same) -> concept = 0. Gap must be ~0."""
    rng = np.random.default_rng(seed)
    Xe = rng.normal(0, 1, (n, d)); Xl = rng.normal(0, 1, (n, d))   # identical P(x)
    ye = (1.0 * Xe[:, 0] + rng.normal(0, 1, n) > 0).astype(int)    # shallow logit (noisy)
    yl = (6.0 * Xl[:, 0] + rng.normal(0, 1, n) > 0).astype(int)    # steep logit (clean)
    return Xe, ye, Xl, yl


def synth_prior_shift(n=8000, d=6, seed=0):
    """Fixed P(y|x)=σ(3 x0); P(y) shifted by a threshold offset early vs late. concept=0."""
    rng = np.random.default_rng(seed)
    Xe = rng.normal(0, 1, (n, d)); Xl = rng.normal(0, 1, (n, d))
    pe = 1 / (1 + np.exp(-(3.0 * Xe[:, 0] - 0.8)))   # same conditional shape, offset
    pl = 1 / (1 + np.exp(-(3.0 * Xl[:, 0] + 0.8)))   # -> different P(y), same ranking rule
    ye = (rng.random(n) < pe).astype(int); yl = (rng.random(n) < pl).astype(int)
    return Xe, ye, Xl, yl


def synth_real_concept(n=8000, d=6, seed=0):
    """Positive control: rule flips early->late (true concept). Gap must be LARGE."""
    rng = np.random.default_rng(seed)
    Xe = rng.normal(0, 1, (n, d)); Xl = rng.normal(0, 1, (n, d))
    ye = (3.0 * Xe[:, 0] + rng.normal(0, 0.4, n) > 0).astype(int)
    yl = (-3.0 * Xl[:, 0] + rng.normal(0, 0.4, n) > 0).astype(int)   # flipped
    return Xe, ye, Xl, yl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--elec2", action="store_true")
    ap.add_argument("--insects", action="store_true")
    ap.add_argument("--insects-variant", default="incremental_balanced")
    ap.add_argument("--tabred", nargs="*", default=[], help="TabReD dataset names")
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--synth-only", action="store_true")
    args = ap.parse_args()
    out_dir = Path("results/phase1/gap_controls"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---- synthetic nulls + positive control (run anywhere) ----
    print("\n==== SYNTHETIC controls (true concept known) ====")
    print(f"  {'case':22s}{'true_gap':>10s}{'placebo':>10s}{'bias_corr':>10s}  expect")
    for nm, gen, expect in [("real_concept(+ctrl)", synth_real_concept, "LARGE>0"),
                            ("noise_drift_null", synth_noise_drift, "~0"),
                            ("prior_shift_null", synth_prior_shift, "~0")]:
        accs_t, accs_p = [], []
        for s in range(min(args.n_seeds, 8)):
            Xe, ye, Xl, yl = gen(seed=s)
            gt = _gap(Xe, ye, Xl, yl, "binclass", s, False)
            gp = _gap(Xe, ye, Xl, yl, "binclass", s, True)
            if gt is not None:
                accs_t.append(gt)
            if gp is not None:
                accs_p.append(gp)
        mt = float(np.mean(accs_t)) if accs_t else float("nan")
        mp = float(np.mean(accs_p)) if accs_p else float("nan")
        print(f"  {nm:22s}{mt:+10.4f}{mp:+10.4f}{mt-mp:+10.4f}  {expect}")
        rows.append({"synthetic": nm, "true_gap": mt, "placebo": mp, "expect": expect})
    print("  READ: noise/prior nulls must be ~0 (true_gap AND bias_corr). If noise_drift")
    print("  fires like a real concept, the estimator confounds Bayes-noise drift => Elec2")
    print("  +0.132 is suspect. real_concept must be large + survive (placebo ~0).")

    if args.synth_only:
        (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2, default=float))
        print(f"\n  wrote {out_dir}/summary.json"); return

    # ---- real data: true gap vs placebo null ----
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

    print(f"\n==== REAL data: true gap vs PLACEBO null ({args.n_seeds} seeds) ====")
    print(f"  {'dataset':24s}{'true_gap [CI]':>26s}{'placebo [CI]':>26s}{'corrected':>11s}  survives?")
    for name, data in jobs:
        t = data.train.t; med = float(np.median(t))
        X = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
        em, lm = t <= med, t > med
        r = real_controls(name, X[em], data.train.y[em], X[lm], data.train.y[lm],
                          data.task, args.n_seeds)
        rows.append(r)
        tci = r["true_gap_ci"]; pci = r["placebo_ci"]
        tg = f"{r['true_gap_mean']:+.3f}[{tci[0]:+.3f},{tci[1]:+.3f}]" if r["true_gap_mean"] is not None else "  n/a"
        pg = f"{r['placebo_mean']:+.3f}[{pci[0]:+.3f},{pci[1]:+.3f}]" if r["placebo_mean"] is not None else "  n/a"
        print(f"  {name:24s}{tg:>26s}{pg:>26s}{r['bias_corrected_gap']:+11.4f}  "
              f"{'YES' if r['survives_placebo'] else '** NO **'}")
    print("\n  GATE: a positive concept claim survives iff true-gap CI lies ABOVE placebo CI.")
    print("  'NO' => that dataset's gap is home-field/N-asymmetry, not concept (PLAN_V3 §0.1).")
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2, default=float))
    print(f"\n  wrote {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
