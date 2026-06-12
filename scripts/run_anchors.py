"""Anchor baselines for Q2b — external calibration of the arm comparison (V2, R0.3-12).

The Q2b 3/5-arm comparison is internally fair but externally uncalibrated: without
reference points a reviewer cannot tell whether ALL arms sit below a trivial
baseline (audit 2026-06-12, confound #5/#6). This script runs, on the SAME
splits/features as the neural arms:

  - lgbm / lgbm_t   : LightGBM without / with the time feature (the standard
                      strong tabular baseline; skipped if lightgbm not installed)
  - knn / knn_t     : k-NN without / with the time feature (k tuned on val from
                      {5, 25, 100}) — the non-parametric lower anchor
  - no_change       : persistence (predict the PREVIOUS sample's label in stream
                      order). On Elec2 this is famously strong (~85%; Žliobaitė
                      2013) — if it beats the neural arms, the whole comparison
                      sits below a trivial baseline and must be reported as such.

Appends one record (mode='anchors') to the same diagnostics.jsonl the Q2b runner
uses, so anchors and arms live in one ledger.

    python scripts/run_anchors.py --dataset elec2 --split temporal
    python scripts/run_anchors.py --dataset insects --insects-variant incremental_balanced
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data.elec2_loader import load_elec2  # noqa: E402
from src.data.insects_loader import load_insects  # noqa: E402
from src.training.tabr_trainer import _build_features  # noqa: E402
from src.utils.metrics import compute_metric, metric_name  # noqa: E402


def _load(args, split, seed):
    if args.dataset == "insects":
        return load_insects(variant=args.insects_variant, split=split, seed=seed,
                            max_samples=args.max_samples)
    return load_elec2(split=split, seed=seed)


def _score(task, y_true, model, X):
    """Primary metric for a fitted sklearn-API model on X."""
    if task == "binclass":
        return compute_metric(y_true, model.predict_proba(X)[:, 1], task)
    return compute_metric(y_true, model.predict(X), task)


def run_knn(task, X, y, with_t, t):
    """k-NN, k tuned on val; standardized features (+ t column when with_t)."""
    def feats(p):
        f = X[p]
        if with_t:
            f = np.concatenate([f, t[p].reshape(-1, 1)], axis=1)
        return f
    sc = StandardScaler().fit(feats("train"))
    Xtr, Xva, Xte = (sc.transform(feats(p)) for p in ("train", "val", "test"))
    best_k, best_val, best_model = None, -np.inf, None
    for k in (5, 25, 100):
        m = KNeighborsClassifier(n_neighbors=min(k, len(y["train"]))).fit(Xtr, y["train"])
        v = _score(task, y["val"], m, Xva)
        if v > best_val:
            best_k, best_val, best_model = k, v, m
    return {"test": _score(task, y["test"], best_model, Xte),
            "val": float(best_val), "k": best_k}


def run_lgbm(task, X, y, with_t, t, seed):
    """LightGBM with val early stopping; returns None if lightgbm unavailable."""
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    def feats(p):
        f = X[p]
        if with_t:
            f = np.concatenate([f, t[p].reshape(-1, 1)], axis=1)
        return f
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31,
                           random_state=seed, verbose=-1)
    m.fit(feats("train"), y["train"],
          eval_set=[(feats("val"), y["val"])],
          eval_metric="auc" if task == "binclass" else "multi_logloss",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return {"test": _score(task, y["test"], m, feats("test")),
            "val": _score(task, y["val"], m, feats("val")),
            "best_iter": int(getattr(m, "best_iteration_", 0) or 0)}


def run_no_change(task, data):
    """Persistence: predict the previous sample's label in STREAM order.

    Uses t_raw (the global stream index each loader carries) to reconstruct the
    full stream regardless of the split, then predicts y[t_raw-1] for each test
    sample. The standard no-change baseline of the streams literature.
    """
    parts = [data.train, data.val, data.test]
    n_all = int(max(p.t_raw.max() for p in parts)) + 1
    y_full = np.full(n_all, -1, dtype="int64")
    for p in parts:
        y_full[p.t_raw] = p.y
    r = data.test.t_raw
    ok = r > 0
    prev = y_full[r[ok] - 1]
    valid = prev >= 0
    y_true = data.test.y[ok][valid]
    y_prev = prev[valid]
    acc = float((y_true == y_prev).mean())
    out = {"test_acc": acc, "n_eval": int(valid.sum())}
    if task == "binclass":   # AUC with the (hard 0/1) previous label as the score
        out["test_auc"] = float(roc_auc_score(y_true, y_prev))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--dataset", default="elec2", choices=["elec2", "insects"])
    ap.add_argument("--insects-variant", default="incremental_balanced")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--split", default="temporal", choices=["temporal", "random"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    data = _load(args, args.split, args.seed)
    task = data.task
    metric = metric_name(task)
    feats, _ = _build_features(data)   # same representation as the neural arms
    t = {p: getattr(data, p).t for p in ("train", "val", "test")}
    y = {p: getattr(data, p).y for p in ("train", "val", "test")}

    print(f"\n==== ANCHORS [{data.name} / {args.split}]  metric={metric} ====")
    rows = {}

    for name, with_t in (("knn", False), ("knn_t", True)):
        r = run_knn(task, feats, y, with_t, t)
        rows[name] = r
        print(f"  {name:10s} test={r['test']:.4f}  (val={r['val']:.4f}, k={r['k']})")

    for name, with_t in (("lgbm", False), ("lgbm_t", True)):
        r = run_lgbm(task, feats, y, with_t, t, args.seed)
        if r is None:
            print(f"  {name:10s} SKIPPED (pip install lightgbm)")
            continue
        rows[name] = r
        print(f"  {name:10s} test={r['test']:.4f}  (val={r['val']:.4f}, "
              f"iters={r['best_iter']})")

    r = run_no_change(task, data)
    rows["no_change"] = r
    extra = f", auc={r['test_auc']:.4f}" if "test_auc" in r else ""
    print(f"  {'no_change':10s} test_acc={r['test_acc']:.4f}{extra}  (n={r['n_eval']})")
    print("\n  READ: if no_change >= the neural arms, the arm comparison sits below a")
    print("  trivial baseline (Elec2 critique, Žliobaitė 2013) — must be reported.")

    out_dir = Path(cfg.experiment.results_dir).parent / f"{args.dataset}_q2"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"mode": "anchors", "ts": time.time(), "dataset": args.dataset,
           "insects_variant": args.insects_variant, "max_samples": args.max_samples,
           "split": args.split, "seed": args.seed, "metric": metric, "rows": rows}
    with open(out_dir / "diagnostics.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"\n  appended to {out_dir / 'diagnostics.jsonl'}")


if __name__ == "__main__":
    main()
