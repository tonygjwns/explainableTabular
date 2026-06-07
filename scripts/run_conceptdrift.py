"""Measure CONCEPT drift per dataset: does P(y|x) change over time?

Decisive for strategy after the synthetic control proved the mechanism works:
- train-early vs train-late, both evaluated on the held-out FUTURE (test) set.
- large gap (late >> early on the future) => the predictive rule moved
  (concept drift) => a place a working time-indexed memory could help.
- ~0 gap => early rule still holds => covariate shift only => no room for the
  mechanism to improve prediction on this dataset.

    python scripts/run_conceptdrift.py --config configs/phase1.yaml          # 4 sanity sets
    python scripts/run_conceptdrift.py --config configs/phase1.yaml --all    # all 8
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
from src.analysis.drift_measure import _stack, concept_drift_gap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    root = Path(cfg.data.root)
    if args.dataset:
        datasets = [args.dataset]
    elif args.all:
        datasets = list(TABRED_DATASETS)
    else:
        datasets = list(cfg.data.sanity_datasets)
    out_dir = Path(cfg.experiment.results_dir).parent / "conceptdrift"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in datasets:
        data = load_tabred(ds, root, split=cfg.experiment.split)
        Xtr = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
        Xte = _stack(data.test.X_num, data.test.X_bin, data.test.X_cat)
        t = data.train.t
        med = float(np.median(t))
        em, lm = t <= med, t > med
        res = concept_drift_gap(Xtr[em], data.train.y[em], Xtr[lm], data.train.y[lm],
                                Xte, data.test.y, data.task, seed=args.seed)
        res.update({"dataset": ds, "task": data.task})
        rows.append(res)
        (out_dir / f"{ds}.json").write_text(json.dumps(res, indent=2))
        print(f"[{ds:20s}] {res['metric']}: early→fut={res['score_early_on_future']:.4f}  "
              f"late→fut={res['score_late_on_future']:.4f}  "
              f"concept_gap={res['gap_concept']:+.4f} (rel={res['gap_rel']:+.2%})")

    (out_dir / "conceptdrift_summary.json").write_text(
        json.dumps({"datasets": datasets, "rows": rows}, indent=2))
    print("\n==== Concept-drift summary ====")
    print("  gap≈0 => early rule still holds on the future = NO concept drift")
    print("         (covariate shift only -> time-indexed memory can't help prediction)")
    print("  gap large(+) => rule moved = concept drift = room for the mechanism")
    print(f"  saved -> {out_dir/'conceptdrift_summary.json'}")


if __name__ == "__main__":
    main()
