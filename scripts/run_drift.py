"""G3 runner: quantify temporal drift per dataset (model-light, objective).

For each dataset:
  - covariate shift AUC, TRAIN vs TEST (the temporal split's P(x) shift)
  - covariate shift AUC, EARLY vs LATE within train (median-time split)
  - label drift: target pos-rate/mean per time decile + Spearman(t, y)

Interprets the Phase-1 nulls: high covariate/label drift + our method still flat
=> method critique; low drift => "no signal to capture" (wrong stage for the bet).

    python scripts/run_drift.py --config configs/phase1.yaml            # 4 sanity sets
    python scripts/run_drift.py --config configs/phase1.yaml --all      # all 8
    python scripts/run_drift.py --config configs/phase1.yaml --dataset homecredit_default
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
from src.analysis.drift_measure import _stack, covariate_shift_auc, label_drift  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true", help="all 8 datasets")
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
    out_dir = Path(cfg.experiment.results_dir).parent / "drift"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in datasets:
        data = load_tabred(ds, root, split=cfg.experiment.split)
        Xtr = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
        Xte = _stack(data.test.X_num, data.test.X_bin, data.test.X_cat)

        cov_tt = covariate_shift_auc(Xtr, Xte, seed=args.seed)
        # early vs late within train (median normalized time)
        t = data.train.t
        med = float(np.median(t))
        e, l = Xtr[t <= med], Xtr[t > med]
        cov_el = covariate_shift_auc(e, l, seed=args.seed)
        # label drift over time (use all splits concatenated)
        y_all = np.concatenate([data.train.y, data.val.y, data.test.y])
        t_all = np.concatenate([data.train.t, data.val.t, data.test.t])
        lab = label_drift(y_all, t_all, data.task)

        rec = {
            "dataset": ds, "task": data.task,
            "covariate_auc_train_vs_test": cov_tt["auc"],
            "covariate_auc_drop_top5": cov_tt.get("auc_drop_top5"),
            "max_single_feat_auc": cov_tt.get("max_single_feat_auc"),
            "n_feat_auc_gt_0.9": cov_tt.get("n_feat_auc_gt_0.9"),
            "covariate_auc_early_vs_late": cov_el["auc"],
            "label_spearman_t_y": lab["spearman_t_y"],
            "label_rel_range_deciles": lab["rel_range"],
            "label_per_bin": lab["per_bin"],
        }
        rows.append(rec)
        (out_dir / f"{ds}.json").write_text(json.dumps({**rec, "cov_tt": cov_tt,
                                                        "cov_el": cov_el, "label": lab}, indent=2))
        print(f"[{ds:20s}] cov(t→te)={rec['covariate_auc_train_vs_test']:.3f} "
              f"drop5={rec['covariate_auc_drop_top5']:.3f} "
              f"maxSF={rec['max_single_feat_auc']:.3f} "
              f"nSF>.9={rec['n_feat_auc_gt_0.9']:<3d} "
              f"| label ρ={rec['label_spearman_t_y']:+.3f} relrange={rec['label_rel_range_deciles']:.3f}")

    (out_dir / "drift_summary.json").write_text(json.dumps({"datasets": datasets, "rows": rows}, indent=2))
    print("\n==== Drift summary ====")
    print("  covariate AUC ~0.5 => no P(x) shift; ->1.0 => strong feature drift.")
    print("  |ρ(t,y)| large or label rel-range large => target/prior drift over time.")
    print(f"  saved -> {out_dir/'drift_summary.json'}")


if __name__ == "__main__":
    main()
