"""R2.2 — DISDE degeneration vs the within-overlap frame (Claim A's raison d'être).

Lead-claim engine: shows that the very covariate dominance defining TabReD-style
temporal drift makes the DISDE (Cai/Namkoong/Yadlowsky, Operations Research 2025)
importance-reweighting estimate of the within-support Y|X term DEGENERATE (ESS
collapse / heavy-tailed weights), while our within-overlap model-transfer frame
(covariate-matched band, no global reweight) stays usable — or, where covariate
shift is so strong the band is empty, certifies concept as TRULY unmeasurable.

Per dataset (early/late by median train t) it joins three VERIFIED measurements:
  1. covariate_shift_auc      — strength of P(x) shift (early-vs-late HGB AUC,
                                + drop-top5 pervasiveness, single-feat proxy)
  2. disde_iw_degeneration    — DISDE-style density-ratio reweighting health:
                                ESS / ESS% / CV(w) / max single-point weight share
  3. concept_within_overlap   — our frame: transfer gap on a fixed late-overlap
                                test (concept, not difficulty) + p-strata stability
                                + n_overlap per half (0 => no common support).

Verdict per dataset:
  DISDE-degenerate + overlap usable  => OUR FRAME WINS (measurable on common support)
  DISDE-degenerate + no overlap      => concept TRULY UNMEASURABLE (covariate destroys support)
  DISDE healthy (low cov AUC)        => both usable (typically concept≈0 there)

    python scripts/run_disde_degeneration.py --config configs/phase1.yaml --all --elec2
    python scripts/run_disde_degeneration.py --config configs/phase1.yaml --all --elec2 --insects

Model-light (sklearn HGB only; no GPU/TabM). Reuses src/analysis/drift_measure.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.data.tabred_loader import load_tabred, TABRED_DATASETS  # noqa: E402
from src.analysis.drift_measure import (  # noqa: E402
    _stack, covariate_shift_auc, disde_iw_degeneration, concept_within_overlap,
)

# thresholds (screen-level; see F3). Two degeneration modes:
#   OVERLAP_MASS_FLOOR: below this, early/late supports are DISJOINT (bias mode).
#   ESS_PCT_FLOOR: overlap exists but weights are heavy-tailed (variance mode).
COV_AUC_HI, ESS_PCT_FLOOR, OVERLAP_MASS_FLOOR, OVERLAP_MIN = 0.90, 5.0, 0.05, 200


def _early_late(data):
    t = data.train.t; med = float(np.median(t))
    X = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
    em, lm = t <= med, t > med
    return X[em], data.train.y[em], X[lm], data.train.y[lm]


def _verdict(cov_auc, overlap_mass, ess_pct, n_ov_min):
    disjoint = overlap_mass < OVERLAP_MASS_FLOOR          # bias mode (disjoint support)
    heavy_tail = (not disjoint) and (ess_pct < ESS_PCT_FLOOR)   # variance mode
    disde_degen = disjoint or heavy_tail or (cov_auc >= COV_AUC_HI)
    mode = ("disjoint-support" if disjoint
            else "heavy-tail" if heavy_tail else "")
    overlap_usable = n_ov_min is not None and n_ov_min >= OVERLAP_MIN
    if disde_degen and overlap_usable:
        return f"DISDE-degenerate[{mode or 'cov'}]; OUR-FRAME-WINS (measurable on common support)"
    if disde_degen and not overlap_usable:
        return f"DISDE-degenerate[{mode or 'cov'}]; concept TRULY UNMEASURABLE (no common support)"
    if overlap_usable:
        return "DISDE-ok; both usable (check gap≈0 => no concept where measurable)"
    return "low-shift / inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--all", action="store_true", help="all 8 TabReD datasets")
    ap.add_argument("--elec2", action="store_true")
    ap.add_argument("--insects", action="store_true",
                    help="add an INSECTS variant (needs river)")
    ap.add_argument("--insects-variant", default="incremental_balanced")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    root = Path(cfg.data.root)
    out_dir = Path(cfg.experiment.results_dir).parent / "disde_degeneration"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    names = list(TABRED_DATASETS) if args.all else list(cfg.data.sanity_datasets)
    for ds in names:
        d = load_tabred(ds, root, split=cfg.experiment.split)
        jobs.append((ds, d.task, _early_late(d)))
    if args.elec2:
        from src.data.elec2_loader import load_elec2
        d = load_elec2(split="temporal", seed=args.seed)
        jobs.append(("elec2", d.task, _early_late(d)))
    if args.insects:
        from src.data.insects_loader import load_insects
        d = load_insects(variant=args.insects_variant, split="temporal", seed=args.seed)
        jobs.append((f"insects_{args.insects_variant}", d.task, _early_late(d)))

    print(f"\n==== DISDE degeneration vs within-overlap "
          f"(overlap<{OVERLAP_MASS_FLOOR}=disjoint | ESS%<{ESS_PCT_FLOOR}=heavy-tail "
          f"| cov_AUC>={COV_AUC_HI}) ====")
    hdr = (f"  {'dataset':22s} {'cov_AUC':>7s} {'drop5':>6s} | "
           f"{'ovlap':>6s} {'ESS%':>6s} {'CV(w)':>7s} {'maxw':>6s} | "
           f"{'n_ov':>6s} {'gap':>8s} | verdict")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows = []
    for ds, task, (Xe, ye, Xl, yl) in jobs:
        cov = covariate_shift_auc(Xe, Xl, seed=args.seed)
        deg = disde_iw_degeneration(Xe, Xl, seed=args.seed)
        con = concept_within_overlap(Xe, ye, Xl, yl, task, seed=args.seed)
        n_ov_min = (min(con.get("n_overlap_early", 0), con.get("n_overlap_late", 0))
                    if con.get("measurable") else 0)
        gap = con.get("concept_gap_within_overlap")
        v = _verdict(cov.get("auc", 0.5), deg.get("overlap_mass", 0.0),
                     deg.get("ess_pct", 100.0), n_ov_min)
        row = {"dataset": ds, "task": task,
               "cov_auc": cov.get("auc"), "cov_auc_drop_top5": cov.get("auc_drop_top5"),
               "disde": deg, "within_overlap": con, "n_overlap_min": n_ov_min,
               "concept_gap": gap, "verdict": v}
        rows.append(row)
        gtxt = f"{gap:+.3f}" if isinstance(gap, (int, float)) else "   -  "
        print(f"  {ds:22s} {cov.get('auc',0.5):7.3f} "
              f"{cov.get('auc_drop_top5',float('nan')):6.3f} | "
              f"{deg.get('overlap_mass',float('nan')):6.3f} "
              f"{deg.get('ess_pct',float('nan')):6.2f} {deg.get('cv',float('nan')):7.1f} "
              f"{deg.get('max_weight_share',float('nan')):6.3f} | "
              f"{n_ov_min:6d} {gtxt:>8s} | {v}")
        (out_dir / f"{ds}.json").write_text(json.dumps(row, indent=2, default=float))

    (out_dir / "summary.json").write_text(json.dumps(
        {"thresholds": {"cov_auc_hi": COV_AUC_HI, "ess_pct_floor": ESS_PCT_FLOOR,
                        "overlap_mass_floor": OVERLAP_MASS_FLOOR,
                        "overlap_min": OVERLAP_MIN}, "rows": rows}, indent=2, default=float))
    print("\n  READ (Claim A): high cov_AUC (pervasive: drop5 still high) => ESS% collapses")
    print("  => DISDE within-support term unestimable; yet where n_ov>=200 our transfer-gap")
    print("  frame still measures concept on common support (the win). n_ov=0 => truly")
    print("  unmeasurable. -> 'covariate dominance makes concept unmeasurable by the")
    print("  standard conditional/reweighting lens; within-overlap recovers it where support exists.'")
    print(f"\n  wrote {out_dir}/  <-- send me summary.json")


if __name__ == "__main__":
    main()
