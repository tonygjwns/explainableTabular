"""Within-overlap concept measurement (spine fix for the F3 elec2 contradiction).

F3 flagged elec2 as overlap 0.438 / ESS 20 — contradictory. ESS=20 is a global-IW
heavy-tail artifact, not 'no common support'. This re-measures concept WITHIN the
overlap band (covariate matched, no global reweight): does P(y|x) move early→late
inside the common-support region?

  python scripts/run_concept_overlap.py --config configs/phase1.yaml --all --elec2

Reads: per dataset, n_overlap/half, early→late vs late→late score, concept_gap.
gap≈0 inside overlap => no concept even where covariate overlaps; gap large =>
real concept on common support (Q2 has a benchmark). Decisive for the §6(다) pivot.
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
from src.analysis.drift_measure import _stack, concept_within_overlap  # noqa: E402


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
    out_dir = Path(cfg.experiment.results_dir).parent / "concept_overlap"
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

    rows = []
    for ds, task, (Xe, ye, Xl, yl) in jobs:
        r = concept_within_overlap(Xe, ye, Xl, yl, task, seed=args.seed)
        r.update({"dataset": ds, "task": task})
        rows.append(r)
        if r.get("measurable"):
            print(f"[{ds:20s}] {r['metric']}: early→lateOv={r['score_early_on_lateOverlap']:.4f} "
                  f"late→lateOv={r['score_late_on_lateOverlap']:.4f} "
                  f"concept_gap={r['concept_gap_within_overlap']:+.4f} "
                  f"(n_ov e/l={r['n_overlap_early']}/{r['n_overlap_late']})")
        else:
            print(f"[{ds:20s}] not measurable within overlap ({r.get('note','')}; "
                  f"n_ov e/l={r.get('n_overlap_early','?')}/{r.get('n_overlap_late','?')})")
        (out_dir / f"{ds}.json").write_text(json.dumps(r, indent=2))

    (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2))
    print("\n==== within-overlap concept ====")
    print("  gap≈0 in overlap => no concept even on common support;")
    print("  gap large => real concept on common support => Q2 has a benchmark there.")


if __name__ == "__main__":
    main()
