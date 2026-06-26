"""V3.5 — is the diagnosis GENERATIVE? Does a method correct under our assumption win
exactly where we MEASURE concept, and not where we don't? (the user's falsification test)

Our account says: temporal tabular shift is covariate-dominated, and where concept DOES
exist the right inductive bias is ONLINE / RECENCY adaptation (a drift prior), NOT a static
concept-targeting structure (which is why our time-indexed retrieval failed — wrong bias).
A correct diagnosis must be GENERATIVE: a recency-adapting model should beat a static model
*only* on datasets where we measure concept (INSECTS, maybe de-time-leaked Elec2), and tie
or lose where concept ≈ 0 (most TabReD, cooking/maps). If the win pattern tracks the measured
concept gap, the frame predicts what works; if it does not, the frame is merely descriptive.

Model-light (sklearn HGB; CPU). Per dataset & representation (full / de-time-leaked):
  - static_all       : HGB on ALL train  -> temporal test   (the no-adaptation baseline)
  - recent_W         : HGB on the most recent W of train (W∈{.25,.5}) -> test (forget old)
  - recency_weighted : HGB with exp recency sample-weights -> test
  recency_gain = best(recency methods) − static_all   (oriented higher=better)
  concept_gap  = ess-gated within-overlap gap on the train early/late split (our measure)

Then ACROSS datasets: Spearman(recency_gain, concept_gap). Pre-registered read:
  - POSITIVE & recency wins on the concept sets, ~0 on concept≈0 sets => diagnosis GENERATIVE
    (the frame predicts where adaptation pays) — the affirmative result the paper needs.
  - ~0 / no pattern => the frame is descriptive only; concept, even where measured, is not
    exploitable by recency => honest pivot signal.

    python scripts/run_correct_assumption.py --tabred cooking_time maps_routing sberbank_housing \
        homecredit_default ecom_offers homesite_insurance weather --elec2 --insects
    python scripts/run_correct_assumption.py --synth-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402

from src.analysis.drift_measure import _stack, concept_within_overlap  # noqa: E402
from src.utils.metrics import compute_metric, metric_name  # noqa: E402


def orient_higher_is_better(scores, metric):
    """Flip sign for lower-is-better metrics so larger = better (avoids importing
    stats.py, which pulls statsmodels)."""
    s = np.asarray(scores, dtype=float)
    return -s if metric.lower() in {"rmse", "mae", "mse", "logloss", "log_loss"} else s

TIMEPROXY_CORR = 0.30
NOISE_FLOOR = 0.034


def _timeproxy_mask(X, t):
    t = np.asarray(t, float); leak = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        c = X[:, j].astype(float); m = ~np.isnan(c)
        if m.sum() < 10 or np.nanstd(c) == 0:
            continue
        r = np.corrcoef(c[m], t[m])[0, 1]
        leak[j] = abs(r) > TIMEPROXY_CORR if np.isfinite(r) else False
    return leak


def _ci95(a):
    a = np.asarray(a, float)
    if len(a) < 2:
        return [float(a[0]), float(a[0])] if len(a) else [float("nan"), float("nan")]
    m, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
    return [float(m - 1.96 * se), float(m + 1.96 * se)]


def _fit_score(Xtr, ytr, Xte, yte, task, seed, sample_weight=None):
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
    if task == "binclass":
        return compute_metric(yte, m.predict_proba(Xte)[:, 1], task)
    return compute_metric(yte, m.predict(Xte), task)


def methods(Xtr, ttr, ytr, Xte, yte, task, seed):
    order = np.argsort(ttr, kind="stable")
    n = len(ytr)
    out = {"static_all": _fit_score(Xtr, ytr, Xte, yte, task, seed)}
    for W in (0.25, 0.5):
        idx = order[-max(int(W * n), 50):]
        out[f"recent_{int(W*100)}"] = _fit_score(Xtr[idx], ytr[idx], Xte, yte, task, seed)
    tn = (ttr - ttr.min()) / (ttr.max() - ttr.min() + 1e-12)
    w = np.exp(3.0 * (tn - 1.0))                       # recent≈1, old≈exp(-3)≈0.05
    out["recency_weighted"] = _fit_score(Xtr, ytr, Xte, yte, task, seed, sample_weight=w)
    return out


def eval_rep(Xtr, ttr, ytr, Xte, yte, task, seeds):
    """Returns recency_gain mean+CI and the per-method means over seeds."""
    metric = metric_name(task)
    rows = {k: [] for k in ("static_all", "recent_25", "recent_50", "recency_weighted")}
    gains = []
    for s in seeds:
        m = methods(Xtr, ttr, ytr, Xte, yte, task, s)
        if m["static_all"] is None:
            continue
        for k in rows:
            if m.get(k) is not None:
                rows[k].append(m[k])
        rec = [m[k] for k in ("recent_25", "recent_50", "recency_weighted") if m.get(k) is not None]
        if rec:
            best_rec = orient_higher_is_better(rec, metric).max()
            stat = float(orient_higher_is_better([m["static_all"]], metric)[0])
            gains.append(float(best_rec - stat))
    means = {k: (float(np.mean(v)) if v else None) for k, v in rows.items()}
    return {"metric": metric, "method_means": means,
            "recency_gain_mean": (float(np.mean(gains)) if gains else None),
            "recency_gain_ci": (_ci95(gains) if gains else None),
            "n_seeds": len(gains)}


def _concept_gap(Xtr, ttr, ytr, task, seed):
    med = float(np.median(ttr)); em, lm = ttr <= med, ttr > med
    r = concept_within_overlap(Xtr[em], ytr[em], Xtr[lm], ytr[lm], task, seed=seed)
    return (r.get("concept_gap_within_overlap") if r.get("measurable") else None,
            r.get("ess_pct"))


def run_dataset(name, data, seeds, de_time_leak=True):
    Xtr = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
    Xte = _stack(data.test.X_num, data.test.X_bin, data.test.X_cat)
    ttr = np.asarray(data.train.t, float)
    ytr, yte, task = data.train.y, data.test.y, data.task
    reps = {"full": (Xtr, Xte)}
    if de_time_leak:
        leak = _timeproxy_mask(Xtr, ttr)
        if leak.any() and (~leak).any():
            reps["de_time_leak"] = (Xtr[:, ~leak], Xte[:, ~leak])
    out = {"dataset": name, "task": task, "reps": {}}
    for rep, (Xa, Xb) in reps.items():
        gap, ess = _concept_gap(Xa, ttr, ytr, task, seeds[0])
        ev = eval_rep(Xa, ttr, ytr, Xb, yte, task, seeds)
        ev["concept_gap"] = gap; ev["ess_pct"] = ess
        out["reps"][rep] = ev
    return out


# ---- synthetic smoke: concept stream (recency should win) vs covariate-only (should not) ----
def _synth(kind, seed, n=6000, d=6):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n)); tr = t < 0.7
    X = rng.normal(0, 1, (n, d))
    if kind == "concept":                       # rule rotates with time -> recency helps
        coef = np.where(t[:, None] < 0.5, 1.0, -1.0)
        y = (3 * coef[:, 0] * X[:, 0] + rng.normal(0, .4, n) > 0).astype(int)
    else:                                        # covariate-only: rule fixed, P(x) drifts
        X[:, 0] += 4 * t
        y = (3 * X[:, 0] - 6 * t + rng.normal(0, .4, n) > 0).astype(int)  # fixed rule in (x,t)
    from types import SimpleNamespace
    mk = lambda m: SimpleNamespace(X_num=X[m], X_bin=None, X_cat=None, y=y[m], t=t[m])
    return SimpleNamespace(train=mk(tr), test=mk(~tr), task="binclass")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--tabred", nargs="*", default=[])
    ap.add_argument("--elec2", action="store_true")
    ap.add_argument("--insects", action="store_true")
    ap.add_argument("--insects-variant", default="incremental_balanced")
    ap.add_argument("--river", nargs="*", default=None,
                    help="river synth streams to include; 'all' = the full panel "
                         "(known-drift breadth for the generative test). e.g. --river all")
    ap.add_argument("--river-n", type=int, default=8000, help="samples per river stream")
    ap.add_argument("--insects-variants", nargs="*", default=None,
                    help="multiple INSECTS variants (more designed-drift breadth)")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synth-only", action="store_true")
    args = ap.parse_args()
    seeds = [args.seed + i for i in range(max(1, args.n_seeds))]
    out_dir = Path("results/phase1/correct_assumption"); out_dir.mkdir(parents=True, exist_ok=True)

    def show(r):
        for rep, ev in r["reps"].items():
            g = ev["concept_gap"]; gt = f"{g:+.3f}" if isinstance(g, (int, float)) else "abstain"
            rg = ev["recency_gain_mean"]
            rgt = (f"{rg:+.4f}{ev['recency_gain_ci']}" if rg is not None else "  -")
            mm = ev["method_means"]
            print(f"    [{rep:12s}] concept={gt:>8s} | static={mm['static_all']} "
                  f"recent50={mm['recent_50']} recW={mm['recency_weighted']} "
                  f"=> recency_gain={rgt}")

    results = []
    if args.synth_only:
        print("\n==== SYNTHETIC smoke (concept stream: recency SHOULD win; covariate-only: should NOT) ====")
        for kind in ("concept", "covariate"):
            r = run_dataset(f"synth_{kind}", _synth(kind, 0), seeds[:3], de_time_leak=False)
            print(f"  synth_{kind}:"); show(r); results.append(r)
        (out_dir / "summary.json").write_text(json.dumps({"results": results}, indent=2, default=float))
        print(f"\n  wrote {out_dir}/summary.json"); return

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config); root = Path(cfg.data.root)
    jobs = []
    for ds in args.tabred:
        from src.data.tabred_loader import load_tabred
        jobs.append((ds, load_tabred(ds, root, split=cfg.experiment.split)))
    if args.elec2:
        from src.data.elec2_loader import load_elec2
        jobs.append(("elec2", load_elec2(split="temporal", seed=0)))
    if args.insects:
        from src.data.insects_loader import load_insects
        jobs.append((f"insects_{args.insects_variant}",
                     load_insects(variant=args.insects_variant, split="temporal", seed=0)))
    for v in (args.insects_variants or []):
        from src.data.insects_loader import load_insects
        jobs.append((f"insects_{v}", load_insects(variant=v, split="temporal", seed=0)))
    if args.river is not None:
        from src.data.river_streams import load_river_stream, list_streams
        names = list_streams(args.river_n) if args.river == ["all"] else args.river
        for nm in names:
            try:
                jobs.append((f"river_{nm}", load_river_stream(nm, n_samples=args.river_n, seed=0)))
            except Exception as e:
                print(f"  SKIP river/{nm}: {type(e).__name__}: {e}")

    print("\n==== GENERATIVE test: does recency-adaptation win where concept is measured? ====")
    pairs = []     # (concept_gap, recency_gain) over measurable reps for the Spearman
    for name, data in jobs:
        r = run_dataset(name, data, seeds)
        results.append(r)
        print(f"\n  [{name}] task={data.task}")
        show(r)
        for rep, ev in r["reps"].items():
            if isinstance(ev["concept_gap"], (int, float)) and ev["recency_gain_mean"] is not None:
                pairs.append((ev["concept_gap"], ev["recency_gain_mean"], f"{name}/{rep}"))
    print("\n  ==== cross-dataset (measurable reps only) ====")
    if len(pairs) >= 3:
        gg = np.array([p[0] for p in pairs]); rg = np.array([p[1] for p in pairs])
        rho, pval = spearmanr(gg, rg)
        print(f"  Spearman(concept_gap, recency_gain) = {rho:+.3f} (p={pval:.3f}), n={len(pairs)}")
        print("  POSITIVE + recency wins on concept sets / ~0 on concept≈0 => diagnosis GENERATIVE.")
        print("  ~0 / no pattern => descriptive only; concept not exploitable by recency => pivot.")
        summary_stat = {"spearman_concept_recencygain": float(rho), "p": float(pval),
                        "n_pairs": len(pairs), "pairs": pairs}
    else:
        print("  too few measurable reps for a cross-dataset correlation.")
        summary_stat = {"note": "too few measurable reps"}
    (out_dir / "summary.json").write_text(json.dumps(
        {"results": results, "cross_dataset": summary_stat}, indent=2, default=float))
    print(f"\n  wrote {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
