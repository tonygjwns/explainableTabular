"""V3.5 path C — does RETRIEVAL beat RECENCY exactly on REOCCURRING drift? (resurrect the
original memory/retrieval idea in its precise niche)

The powered generative test (run_correct_assumption, n=18) falsified the simple law
"concept magnitude predicts recency_gain": recency WINS on monotonic drift but LOSES on
reoccurring drift (an old concept returns; recency discards the matching old data). That is
exactly where retrieval-by-similarity should win — k-NN recalls the old examples that match
the present, regardless of recency. If retrieval beats recency *specifically* on reoccurring
streams (and not on monotonic ones), the frame becomes generative again, richer: the measure
says concept is present, and the DRIFT STRUCTURE says which adaptation to use
(monotonic -> recency / reoccurring -> retrieval). It also relocates the original
time-indexed-retrieval idea to the niche where it is the right tool.

Model-light (sklearn). Per dataset, on the temporal test:
  static_all       : HGB on ALL train                              (no adaptation)
  recency          : best of {HGB on recent 50%, HGB recency-weighted}   (forget old)
  retrieval        : k-NN on ALL train, k tuned on val (recall by similarity, keeps old)
  retrieval_recent : k-NN on recent 50% only (control: retrieval WITHOUT the old data)
gains vs static (oriented higher=better); the key contrast is retrieval_gain − recency_gain,
grouped by drift structure (nodrift / monotonic / reoccurring).

PRE-REGISTERED: retrieval_gain − recency_gain > 0 on REOCCURRING and ≤ 0 on MONOTONIC
(nodrift ~0 both) => retrieval is the right tool for reoccurring drift; the original idea
lives in that niche; "measure concept + diagnose drift structure -> pick adaptation" is the
generative story. A null (retrieval no better than recency on reoccurring) => the memory idea
does not even win on its home field => accept the pivot.

    python scripts/run_retrieval_vs_recency.py --river all --insects-variants \
        incremental_balanced incremental_reoccurring_balanced abrupt_balanced
    python scripts/run_retrieval_vs_recency.py --synth-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402

from src.analysis.drift_measure import _stack, concept_within_overlap  # noqa: E402
from src.utils.metrics import compute_metric, metric_name  # noqa: E402


def orient(scores, metric):
    s = np.asarray(scores, float)
    return -s if metric.lower() in {"rmse", "mae", "mse", "logloss", "log_loss"} else s


def _ci95(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return [float(a[0]), float(a[0])] if len(a) else [float("nan"), float("nan")]
    m, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
    return [float(m - 1.96 * se), float(m + 1.96 * se)]


def _hgb(Xtr, ytr, Xte, yte, task, seed, sample_weight=None):
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    with np.errstate(all="ignore"):
        keep = (~np.all(np.isnan(Xtr), axis=0)) & (np.nanstd(Xtr, axis=0) > 0)
    if keep.any():
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    if task == "regression":
        m = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
    else:
        if len(np.unique(ytr)) < 2:
            return None
        m = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
    m.fit(Xtr, ytr, sample_weight=sample_weight)
    return compute_metric(yte, m.predict_proba(Xte)[:, 1] if task == "binclass" else m.predict(Xte), task)


def _knn(Xtr, ytr, Xva, yva, Xte, yte, task, metric):
    """k-NN (retrieval by similarity); k chosen on val. Imputed+standardized features."""
    if task != "regression" and len(np.unique(ytr)) < 2:
        return None
    best_k, best_v, best = None, -np.inf, None
    for k in (15, 50, 150):
        kk = min(k, len(ytr))
        Knn = KNeighborsRegressor if task == "regression" else KNeighborsClassifier
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Knn(n_neighbors=kk))
        pipe.fit(Xtr, ytr)
        pv = (pipe.predict_proba(Xva)[:, 1] if task == "binclass"
              else pipe.predict(Xva))
        v = float(orient([compute_metric(yva, pv, task)], metric)[0])
        if v > best_v:
            best_v, best_k, best = v, kk, pipe
    pte = best.predict_proba(Xte)[:, 1] if task == "binclass" else best.predict(Xte)
    return compute_metric(yte, pte, task)


def eval_dataset(name, data, kind, seeds):
    Xtr = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
    Xva = _stack(data.val.X_num, data.val.X_bin, data.val.X_cat)
    Xte = _stack(data.test.X_num, data.test.X_bin, data.test.X_cat)
    ttr = np.asarray(data.train.t, float); ytr = data.train.y
    yva, yte, task = data.val.y, data.test.y, data.task
    metric = metric_name(task)
    order = np.argsort(ttr, kind="stable"); nrec = max(int(0.5 * len(ytr)), 50)
    rec_idx = order[-nrec:]
    tn = (ttr - ttr.min()) / (ttr.max() - ttr.min() + 1e-12); w = np.exp(3.0 * (tn - 1.0))

    rg, retg, retrecg = [], [], []          # recency_gain, retrieval_gain, retrieval_recent_gain
    statics = []
    for s in seeds:
        st = _hgb(Xtr, ytr, Xte, yte, task, s)
        if st is None:
            continue
        statics.append(st)
        recency = [v for v in (_hgb(Xtr[rec_idx], ytr[rec_idx], Xte, yte, task, s),
                               _hgb(Xtr, ytr, Xte, yte, task, s, sample_weight=w)) if v is not None]
        ret = _knn(Xtr, ytr, Xva, yva, Xte, yte, task, metric)
        retr = _knn(Xtr[rec_idx], ytr[rec_idx], Xva, yva, Xte, yte, task, metric)
        so = float(orient([st], metric)[0])
        if recency:
            rg.append(float(orient(recency, metric).max() - so))
        if ret is not None:
            retg.append(float(orient([ret], metric)[0] - so))
        if retr is not None:
            retrecg.append(float(orient([retr], metric)[0] - so))
    med = float(np.median(ttr)); em, lm = ttr <= med, ttr > med
    cg = concept_within_overlap(Xtr[em], ytr[em], Xtr[lm], ytr[lm], task, seed=seeds[0])
    rgm = float(np.mean(rg)) if rg else None
    retgm = float(np.mean(retg)) if retg else None
    return {"dataset": name, "kind": kind, "task": task,
            "concept_gap": (cg.get("concept_gap_within_overlap") if cg.get("measurable") else None),
            "recency_gain": rgm, "retrieval_gain": retgm,
            "retrieval_recent_gain": (float(np.mean(retrecg)) if retrecg else None),
            "retrieval_minus_recency": ((retgm - rgm) if (rgm is not None and retgm is not None) else None)}


def _synth(kind, seed, n=6000, d=6):
    """concept A->B->A (reoccurring) vs A->B (monotonic). Test (last 15%) is in A for
    reoccurring (recency trained on recent B should fail; retrieval should recall old A)."""
    rng = np.random.default_rng(seed); t = np.sort(rng.random(n)); X = rng.normal(0, 1, (n, d))
    wA, wB = 1.0, -1.0
    if kind == "reoccurring":
        wt = np.where(t < 0.35, wA, np.where(t < 0.78, wB, wA))
    else:                                   # monotonic A->B
        wt = np.where(t < 0.5, wA, wB)
    y = (3 * wt * X[:, 0] + rng.normal(0, .4, n) > 0).astype(int)
    tr = t < 0.7; va = (t >= 0.7) & (t < 0.85); te = t >= 0.85
    from types import SimpleNamespace
    mk = lambda m: SimpleNamespace(X_num=X[m], X_bin=None, X_cat=None, y=y[m], t=t[m])
    return SimpleNamespace(train=mk(tr), val=mk(va), test=mk(te), task="binclass")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--river", nargs="*", default=None)
    ap.add_argument("--river-n", type=int, default=8000)
    ap.add_argument("--insects-variants", nargs="*", default=None)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synth-only", action="store_true")
    args = ap.parse_args()
    seeds = [args.seed + i for i in range(max(1, args.n_seeds))]
    out_dir = Path("results/phase1/retrieval_vs_recency"); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.synth_only:
        print("\n==== SYNTHETIC smoke (retrieval SHOULD beat recency on reoccurring, not monotonic) ====")
        for kind in ("reoccurring", "monotonic"):
            r = eval_dataset(f"synth_{kind}", _synth(kind, 0), kind, seeds[:3])
            rows.append(r)
            print(f"  synth_{kind:11s} concept={r['concept_gap']} recency={r['recency_gain']:+.4f} "
                  f"retrieval={r['retrieval_gain']:+.4f} => ret-rec={r['retrieval_minus_recency']:+.4f}")
        (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2, default=float))
        print(f"\n  wrote {out_dir}/summary.json"); return

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

    print("\n==== RETRIEVAL vs RECENCY by drift structure ====")
    print(f"  {'dataset':34s}{'kind':12s}{'concept':>8s}{'recency':>9s}{'retr':>9s}{'retr-rec':>9s}")
    for name, data, kind in jobs:
        r = eval_dataset(name, data, kind, seeds)
        rows.append(r)
        def f(x):
            return f"{x:+.3f}" if isinstance(x, (int, float)) else "   -"
        print(f"  {name:34s}{kind:12s}{f(r['concept_gap']):>8s}{f(r['recency_gain']):>9s}"
              f"{f(r['retrieval_gain']):>9s}{f(r['retrieval_minus_recency']):>9s}")

    print("\n  ==== mean(retrieval − recency) by drift structure ====")
    agg = {}
    for kind in ("reoccurring", "monotonic", "nodrift"):
        vals = [r["retrieval_minus_recency"] for r in rows
                if r["kind"] == kind and isinstance(r["retrieval_minus_recency"], (int, float))]
        if vals:
            agg[kind] = {"mean": float(np.mean(vals)), "ci95": _ci95(vals), "n": len(vals)}
            print(f"  {kind:12s} mean(retr−rec) = {np.mean(vals):+.4f} {_ci95(vals)} (n={len(vals)})")
    print("\n  PRE-REGISTERED: retr−rec > 0 on REOCCURRING and <= 0 on MONOTONIC => retrieval is")
    print("  the right tool for reoccurring drift (original memory idea lives there). Null => pivot.")
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "by_kind": agg}, indent=2, default=float))
    print(f"\n  wrote {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
