"""F3 — concept-measurability feasibility probe (PLAN_RESCUE §A, run parallel to Q1).

Under covariate AUC≈1.0 the early/late x supports barely overlap, so "fix x, vary t"
concept measurement may be ILL-POSED ("unmeasurable" ≠ "no concept"). For each
dataset we report overlap mass (band 0.1-0.9 + 0.2-0.8 sensitivity), IW ESS, and
per-time-half label support, then a measurability judgment. If NOTHING is measurable,
Q2's data-gating premise dies -> raise §6(다) weight regardless of Q1.

    python scripts/run_f3_feasibility.py --config configs/phase1.yaml --all   # TabReD 8
    python scripts/run_f3_feasibility.py --config configs/phase1.yaml --elec2  # + Elec2
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
from src.analysis.drift_measure import _stack, overlap_feasibility  # noqa: E402

# measurability thresholds (screen-level; real concept measurement wants ESS 1000-2000)
TAU_M, N_M, EV_FLOOR = 0.05, 500, 30


def _early_late(data):
    t = data.train.t; med = float(np.median(t))
    X = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
    em, lm = t <= med, t > med
    return X[em], data.train.y[em], X[lm], data.train.y[lm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--elec2", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    root = Path(cfg.data.root)
    out_dir = Path(cfg.experiment.results_dir).parent / "f3"; out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    names = list(TABRED_DATASETS) if args.all else list(cfg.data.sanity_datasets)
    for ds in names:
        d = load_tabred(ds, root, split=cfg.experiment.split)
        jobs.append((ds, d.task, _early_late(d)))
    if args.elec2:
        from src.data.elec2_loader import load_elec2
        d = load_elec2(split="temporal", seed=args.seed)   # temporal so early/late by time
        jobs.append(("elec2", d.task, _early_late(d)))

    rows = []
    for ds, task, (Xe, ye, Xl, yl) in jobs:
        r = overlap_feasibility(Xe, ye, Xl, yl, task, seed=args.seed)
        b = r.get("bands", {}).get("0.1-0.9", {})
        mass = b.get("overlap_mass", 0.0); ess = r.get("IW_ESS_early", 0.0)
        ev = min(b.get("minority_events_early", 0), b.get("minority_events_late", 0))
        measurable = bool(mass >= TAU_M and ess >= N_M and ev >= EV_FLOOR)
        rows.append({"dataset": ds, "task": task, "measurable": measurable,
                     "overlap_mass_.1-.9": mass,
                     "overlap_mass_.2-.8": r.get("bands", {}).get("0.2-0.8", {}).get("overlap_mass"),
                     "IW_ESS": ess, "min_minority_events_per_half": ev, "detail": r})
        print(f"[{ds:20s}] overlap(.1-.9)={mass:.3f} (.2-.8)="
              f"{rows[-1]['overlap_mass_.2-.8']:.3f}  ESS={ess:.0f}  "
              f"minEvents/half={ev}  → {'MEASURABLE' if measurable else 'unmeasurable'}")
        (out_dir / f"{ds}.json").write_text(json.dumps(rows[-1], indent=2))

    n_meas = sum(r["measurable"] for r in rows)
    (out_dir / "f3_summary.json").write_text(json.dumps(
        {"thresholds": {"tau_m": TAU_M, "N_m": N_M, "ev_floor": EV_FLOOR},
         "n_measurable": n_meas, "rows": rows}, indent=2))
    print(f"\n==== F3: {n_meas}/{len(rows)} datasets measurable ====")
    print("  0 measurable → Q2 data-gating premise dies → raise §6(다) weight (Q1-independent).")
    print("  measurable set → Q2 dataset pool (common-support).")


if __name__ == "__main__":
    main()
