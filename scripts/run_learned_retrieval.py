"""V3.5-C step 2 — does the LEARNED retrieval structure (TimeTabR) beat a parametric model
AND recency on REOCCURRING drift? (resurrect the original method in its niche)

Step 1 (run_retrieval_vs_recency) showed plain k-NN retrieval beats recency on reoccurring
drift (mean retr−rec +0.192, CI excludes 0) but fails on some feature spaces (agrawal: kNN
weakness). The original contribution is a LEARNED time-indexed retrieval — it should (a) beat
the parametric baseline on reoccurring (recall the reoccurred concept from the full context),
and (b) fix the spaces where plain kNN's fixed metric fails. If so, the dead Claim B flips to
a CONDITIONAL positive: the retrieval STRUCTURE is redundant on monotonic drift (the original
negative) but WINS on reoccurring drift — its identified niche.

Per stream, on the temporal test (eval_context='full' => retrieval sees the OLD reoccurred
examples), train the 5-arm shared-encoder models:
  mlp_t   on ALL train     — parametric + time feature (no retrieval)          [baseline]
  mlp_t   on RECENT 50%    — recency (forget old)                              [recency arm]
  tabr_t  on ALL train     — retrieval structure + time feature               [the structure]
  time_tabr_t on ALL train — retrieval + time hooks (the full method)         [optional]
Gains vs mlp_t(all): retrieval_struct = tabr_t − mlp_t ; recency = mlp_t_recent − mlp_t ;
learned_full = time_tabr_t − mlp_t. Grouped by drift structure (nodrift/monotonic/reoccurring).

PRE-REGISTERED: retrieval_struct > 0 AND > recency on REOCCURRING, and <= 0 (redundant, the
original Claim B negative) on MONOTONIC => the learned retrieval structure is the right tool
for reoccurring drift; Claim B becomes a conditional positive scoped to that niche. A null on
reoccurring => the learned structure does not even win on its home field => accept the pivot.

Neural (GPU). Heavy: keep --river to a subset and --n-seeds small.

    python scripts/run_learned_retrieval.py --river all --n-seeds 3
    python scripts/run_learned_retrieval.py --river stagger_reoccur sine_reoccur2 agrawal_reoccur \
        sea_abrupt stagger_abrupt --insects-variants incremental_reoccurring_balanced incremental_balanced
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.data.tabred_loader import TabularSplit, TabReDDataset  # noqa: E402
from src.training.tabr_trainer import TabRConfig, train_timetabr  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402
from src.utils.metrics import metric_name  # noqa: E402


def _orient(v, metric):
    return -v if metric.lower() in {"rmse", "mae", "mse", "logloss", "log_loss"} else v


def _ci95(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return [float(a[0]), float(a[0])] if len(a) else [float("nan"), float("nan")]
    m, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
    return [float(m - 1.96 * se), float(m + 1.96 * se)]


def _recent_train(data, frac=0.5):
    """A copy of `data` whose TRAIN is the most-recent `frac` of the original train (by t)."""
    tr = data.train
    order = np.argsort(np.asarray(tr.t), kind="stable")
    keep = order[-max(int(frac * len(tr.y)), 50):]
    sub = TabularSplit(
        X_num=None if tr.X_num is None else tr.X_num[keep],
        X_bin=None if tr.X_bin is None else tr.X_bin[keep],
        X_cat=None if tr.X_cat is None else tr.X_cat[keep],
        y=tr.y[keep], t=tr.t[keep], t_raw=tr.t_raw[keep])
    return TabReDDataset(name=data.name + "_recent", task=data.task, split=data.split,
                         train=sub, val=data.val, test=data.test,
                         t_min=float(np.asarray(sub.t).min()), t_max=float(np.asarray(sub.t).max()))


def _cfg(cfg, arch, seed, lr):
    tr = cfg.training
    return TabRConfig(arch=arch, time_basis="fourier",
                      trend_degree=int(OmegaConf.select(cfg, "memory.trend_degree", default=3)),
                      lr=float(lr), weight_decay=1e-4, dropout=0.1,
                      batch_size=int(tr.batch_size), eval_batch=int(tr.eval_batch),
                      patience=int(tr.patience), max_epochs=int(tr.max_epochs),
                      min_epochs=10, seed_tag=f"s{seed}")


def _score(data, arch, seed, lr):
    seed_everything(seed)
    return float(train_timetabr(data, _cfg(cfg_g, arch, seed, lr))["score"])


def eval_stream(name, data, kind, seeds, lr, with_timehook=True):
    metric = metric_name(data.task)
    data_recent = _recent_train(data)
    g_struct, g_rec, g_full = [], [], []
    for s in seeds:
        base = _orient(_score(data, "mlp_t", s, lr), metric)          # parametric, all train
        rec = _orient(_score(data_recent, "mlp_t", s, lr), metric)    # recency
        tab = _orient(_score(data, "tabr_t", s, lr), metric)          # retrieval structure
        g_struct.append(tab - base); g_rec.append(rec - base)
        if with_timehook:
            tt = _orient(_score(data, "time_tabr_t", s, lr), metric)  # full method
            g_full.append(tt - base)
    def m(a):
        return float(np.mean(a)) if a else None
    return {"dataset": name, "kind": kind, "task": data.task, "n_seeds": len(seeds),
            "retrieval_struct_gain": m(g_struct), "recency_gain": m(g_rec),
            "learned_full_gain": (m(g_full) if with_timehook else None),
            "struct_minus_recency": (float(np.mean(g_struct) - np.mean(g_rec)) if g_struct else None)}


cfg_g = None


def main():
    global cfg_g
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--river", nargs="*", default=None, help="stream names or 'all'")
    ap.add_argument("--river-n", type=int, default=8000)
    ap.add_argument("--insects-variants", nargs="*", default=None)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-timehook", action="store_true", help="skip time_tabr_t (cheaper)")
    args = ap.parse_args()
    cfg_g = OmegaConf.load(args.config)
    seeds = [args.seed + i for i in range(max(1, args.n_seeds))]
    out_dir = Path("results/phase1/learned_retrieval"); out_dir.mkdir(parents=True, exist_ok=True)

    from src.data.river_streams import load_river_stream, list_streams, drift_kind
    jobs = []
    if args.river is not None:
        names = list_streams(args.river_n) if args.river == ["all"] else args.river
        for nm in names:
            try:
                jobs.append((f"river_{nm}", load_river_stream(nm, n_samples=args.river_n, seed=0),
                             drift_kind(nm)))
            except Exception as e:
                print(f"  SKIP river/{nm}: {type(e).__name__}: {e}")
    for v in (args.insects_variants or []):
        from src.data.insects_loader import load_insects
        kind = "reoccurring" if "reoccurring" in v else "monotonic"
        jobs.append((f"insects_{v}", load_insects(variant=v, split="temporal", seed=0), kind))

    print("\n==== LEARNED retrieval (tabr_t) vs recency vs parametric (mlp_t), by drift structure ====")
    print(f"  {'dataset':34s}{'kind':12s}{'struct':>9s}{'recency':>9s}{'full':>9s}{'str-rec':>9s}")
    rows = []
    for name, data, kind in jobs:
        r = eval_stream(name, data, kind, seeds, args.lr, with_timehook=not args.no_timehook)
        rows.append(r)
        def f(x):
            return f"{x:+.3f}" if isinstance(x, (int, float)) else "   -"
        print(f"  {name:34s}{kind:12s}{f(r['retrieval_struct_gain']):>9s}{f(r['recency_gain']):>9s}"
              f"{f(r['learned_full_gain']):>9s}{f(r['struct_minus_recency']):>9s}")
        (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2, default=float))

    print("\n  ==== means by drift structure ====")
    agg = {}
    for kind in ("reoccurring", "monotonic", "nodrift"):
        sub = [r for r in rows if r["kind"] == kind]
        sm = [r["struct_minus_recency"] for r in sub if isinstance(r["struct_minus_recency"], (int, float))]
        sg = [r["retrieval_struct_gain"] for r in sub if isinstance(r["retrieval_struct_gain"], (int, float))]
        if sm:
            agg[kind] = {"struct_minus_recency_mean": float(np.mean(sm)), "ci95": _ci95(sm),
                         "retrieval_struct_gain_mean": float(np.mean(sg)), "n": len(sm)}
            print(f"  {kind:12s} struct−recency = {np.mean(sm):+.4f} {_ci95(sm)} | "
                  f"struct_gain = {np.mean(sg):+.4f} (n={len(sm)})")
    print("\n  PRE-REGISTERED: retrieval_struct_gain > 0 AND struct−recency > 0 on REOCCURRING,")
    print("  <= 0 on MONOTONIC (original Claim B negative) => learned retrieval STRUCTURE wins in")
    print("  its reoccurring niche => Claim B flips to a conditional positive. Null => pivot.")
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "by_kind": agg}, indent=2, default=float))
    print(f"\n  wrote {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
