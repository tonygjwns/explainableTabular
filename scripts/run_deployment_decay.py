"""V5 — validate the broad negative with the DEPLOYMENT lens, not the within-overlap lens.

After 8 within-overlap negatives, the earned claim is "unmeasurable by the within-overlap
instrument," NOT "concept is absent." That instrument (median early/late split + common-support
gap) is structurally blind to a changed rule that lives in the NEW, non-overlapping region: it
abstains exactly there (DISDE term-ii, chosen early, never questioned). Malware is the smoking
gun — concept drift is literature-established (TESSERACT) yet our EMBER read concept~0 — pointing
at the tool/split, not the phenomenon.

So we drop that lens and ask the question a deployed model actually faces: train on the PAST,
predict the FUTURE. Three quantities, rolling-origin, imbalance kept:

  decay          : a model trained on the OLDEST window loses how much AUC on future windows
                   (baseline = its own held-out early score).  Necessary, not sufficient (covariate
                   shift alone decays a model).
  recency_gain   : on the SAME future window W_j, a model trained on the most-RECENT past window
                   beats one trained on the OLDEST window (both size-matched).  Controls for window
                   difficulty (the INSECTS achievable-accuracy trap), but NOT for covariate
                   coverage (recent X is simply closer to W_j) -> conflates coverage + rule-change.
  staleness_harm : recent-only(N)  vs  recent+old (the SAME N recent samples + N OLD ones), eval
                   on W_j.  THE concept-isolating discriminator.  The recent portion is IDENTICAL
                   in both, so there is NO covariate-density / coverage confound — the ONLY
                   difference is the N extra OLD samples.  Under a FIXED rule old data carries
                   correct labels -> adding it helps or is neutral -> staleness_harm <= 0.  Under
                   concept drift old labels are WRONG now -> adding them pollutes the fit -> the
                   future score drops -> staleness_harm > 0 = the rule P(y|x) changed = CONCEPT.

PRE-REGISTERED verdict per dataset (binclass: AUC; multiclass: acc; regression: -RMSE):
  DEPLOYMENT-CONCEPT          decay present AND staleness_harm CI>0  -> the within-overlap negative
                              is an INSTRUMENT ARTIFACT; concept is real & exploitable in deployment.
                              (pivot to the deployment-decay frame; overlap lens was blind here.)
  DEPLOYMENT-DECAY-COVARIATE  decay present AND recency_gain CI>0 but staleness_harm ~0/<=0 -> real
                              exploitable temporal structure (recency adaptation recovers it), but
                              the mechanism is covariate coverage, not a changed rule.  Still
                              contradicts "absent"; honest middle.
  DEPLOYMENT-STABLE           no decay, no recency_gain, no staleness_harm -> the negative is
                              hardened on BOTH lenses -> write the broad-negative measurement paper.

Discipline (lessons baked in): size-matched training (no "fewer-samples" confound), same-window
gain comparison (controls window difficulty / INSECTS trap), staleness_harm controls covariate
coverage so a covariate-only decay can't masquerade as concept, multi-seed CIs, and three local
synthetic ground-truth controls that MUST land in the three regimes before any real-data read is
trusted.  Complementary to the autocorrelation guard already in run_adversarial_probe.

Model-light (sklearn). Generic: point it at any time-ordered tabular CSV/parquet.

    python scripts/run_deployment_decay.py --synth                     # ground-truth controls (run FIRST)
    python scripts/run_deployment_decay.py --csv ember.parquet --target label --time appeared --by-value
    python scripts/run_deployment_decay.py --csv baf.parquet --target fraud_bool --time month
    python scripts/run_deployment_decay.py --tabred sberbank_housing homecredit_default --config configs/phase1.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error  # noqa: E402

FLOOR_GAIN = 0.02     # a gain/harm below this (in AUC/acc units) is noise
FLOOR_DECAY = 0.02    # a decay below this is not "the model meaningfully ages"
MIN_POS = 10          # a usable binclass window/sample needs >= this many of the minority class


# ----------------------------------------------------------------------------- scoring
def _fit_score(Xtr, ytr, Xte, yte, task, seed):
    """Higher-is-better score of a fresh HGB trained on (Xtr,ytr), eval on (Xte,yte).
    Returns None if the window can't yield a valid score (single-class train/test, etc.).
    Raw features (HGB handles NaN natively); imbalance is preserved (no resampling)."""
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    if task == "regression":
        m = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(Xtr, ytr)
        return -float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))   # -RMSE: higher better
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    m = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(Xtr, ytr)
    if task == "binclass":
        pos = list(m.classes_).index(m.classes_[-1])
        return float(roc_auc_score(yte, m.predict_proba(Xte)[:, pos]))
    return float(accuracy_score(yte, m.predict(Xte)))


def _ok_train(y, task):
    if task == "regression":
        return len(y) >= 50
    if len(np.unique(y)) < 2:
        return False
    _, c = np.unique(y, return_counts=True)
    return c.min() >= MIN_POS


# ----------------------------------------------------------------------------- windows
def _assign_windows(t, K, by_value):
    """Return per-row window id in [0..K-1] (or fewer). by_value: one window per unique time
    value (e.g. EMBER YYYYMM months = TESSERACT-faithful). else: K equal-count quantile bins."""
    t = np.asarray(t, float)
    if by_value:
        uniq = np.unique(t[np.isfinite(t)])
        remap = {v: i for i, v in enumerate(uniq)}
        return np.array([remap.get(v, -1) for v in t], int), len(uniq)
    ranks = np.argsort(np.argsort(t))                  # 0..n-1 rank, ties broken stably
    w = (ranks * K // len(t)).clip(0, K - 1)
    return w.astype(int), K


def _sample(idx, n, rng):
    if len(idx) <= n:
        return idx
    return rng.choice(idx, n, replace=False)


# ----------------------------------------------------------------------------- core
def _per_seed(X, y, win, task, K, max_train, seed):
    """One seed of the rolling-origin analysis. Returns per-window-averaged
    (decay, recency_gain, staleness_harm) or Nones where not computable."""
    rng = np.random.default_rng(seed)
    by_w = [np.where(win == k)[0] for k in range(K)]
    if not _ok_train(y[by_w[0]], task):
        return None
    # --- oldest-window model for the DECAY curve (its own held-out early baseline) ---
    w0 = rng.permutation(by_w[0]); cut = int(len(w0) * 0.7)
    tr0, ho0 = w0[:cut], w0[cut:]
    tr0 = _sample(tr0, max_train, rng)
    base = _fit_score(X[tr0], y[tr0], X[ho0], y[ho0], task, seed) if _ok_train(y[tr0], task) else None

    back = range(max(1, K // 2), K)                    # evaluate decay/gain on the back half
    decays, recs, stales = [], [], []
    for j in back:
        te = by_w[j]
        if len(te) < 20 or (task != "regression" and len(np.unique(y[te])) < 2):
            continue
        # decay of the oldest model on this future window
        if base is not None:
            s = _fit_score(X[tr0], y[tr0], X[te], y[te], task, seed) if _ok_train(y[tr0], task) else None
            if s is not None:
                decays.append(base - s)
        # recent / old models on the SAME future window, size-matched at N.
        # recency_gain = recent - old. staleness_harm = recent - (recent + old): the recent rows
        # are IDENTICAL in both, so only the N extra OLD rows differ -> no density confound.
        recent_pool, old_pool = by_w[j - 1], by_w[0]
        n = min(len(recent_pool), len(old_pool), max_train)
        if n < 50:
            continue
        recent = _sample(recent_pool, n, rng)
        old = _sample(old_pool, n, rng)
        if not (_ok_train(y[recent], task) and _ok_train(y[old], task)):
            continue
        recent_old = np.concatenate([recent, old])          # same N recent + N old
        s_recent = _fit_score(X[recent], y[recent], X[te], y[te], task, seed)
        s_old = _fit_score(X[old], y[old], X[te], y[te], task, seed)
        s_recent_old = _fit_score(X[recent_old], y[recent_old], X[te], y[te], task, seed)
        if None not in (s_recent, s_old):
            recs.append(s_recent - s_old)
        if None not in (s_recent, s_recent_old):
            stales.append(s_recent - s_recent_old)           # >0 = adding old HURT = concept
    mean = lambda a: float(np.mean(a)) if a else None
    return mean(decays), mean(recs), mean(stales)


def _ci95(a):
    a = np.asarray([v for v in a if v is not None], float)
    if len(a) == 0:
        return None, [None, None]
    if len(a) < 2:
        return float(a[0]), [float(a[0]), float(a[0])]
    m, se = float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))
    return m, [m - 1.96 * se, m + 1.96 * se]


def assess(name, X, y, t, task, K=10, by_value=False, n_seeds=5, max_train=6000):
    win, Keff = _assign_windows(t, K, by_value)
    X, y, win = X[win >= 0], y[win >= 0], win[win >= 0]
    decay_s, rec_s, stale_s = [], [], []
    for s in range(n_seeds):
        out = _per_seed(X, y, win, task, Keff, max_train, s)
        if out is None:
            continue
        d, r, st = out
        decay_s.append(d); rec_s.append(r); stale_s.append(st)
    decay, decay_ci = _ci95(decay_s)
    rec, rec_ci = _ci95(rec_s)
    stale, stale_ci = _ci95(stale_s)

    decay_present = decay is not None and decay > FLOOR_DECAY
    # CI lower bound > 0 AND point above the noise floor
    stale_concept = (stale is not None and stale_ci[0] is not None
                     and stale_ci[0] > 0 and stale > FLOOR_GAIN)
    rec_present = (rec is not None and rec_ci[0] is not None
                   and rec_ci[0] > 0 and rec > FLOOR_GAIN)
    if stale_concept and decay_present:
        verdict = "DEPLOYMENT-CONCEPT"
    elif rec_present and decay_present:
        verdict = "DEPLOYMENT-DECAY-COVARIATE"
    elif not decay_present and not rec_present:
        verdict = "DEPLOYMENT-STABLE"
    else:
        verdict = "INCONCLUSIVE"
    return {"dataset": name, "task": task, "n": int(len(y)), "n_windows": int(Keff),
            "n_seeds_ok": len(decay_s),
            "decay": decay, "decay_ci": decay_ci,
            "recency_gain": rec, "recency_gain_ci": rec_ci,
            "staleness_harm": stale, "staleness_harm_ci": stale_ci,
            "verdict": verdict}


# ----------------------------------------------------------------------------- loaders
def _load_csv(path, target, time, drop, max_n):
    import pandas as pd
    from pandas.api import types as pt
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    df = df.dropna(subset=[target, time])
    if max_n and len(df) > max_n:                       # keep time order; even thinning
        df = df.sort_values(time).iloc[:: max(1, len(df) // max_n)].reset_index(drop=True)

    def to_num(col):
        if pt.is_numeric_dtype(col):
            return pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64")
        if pt.is_datetime64_any_dtype(col):
            return col.view("int64").to_numpy().astype("float64")
        return col.astype("category").cat.codes.to_numpy().astype("float64")

    y_raw = df[target]
    tnum = to_num(df[time])
    drop_cols = set([target, time] + list(drop or []))
    feats = [c for c in df.columns if c not in drop_cols]
    X = np.column_stack([to_num(df[c]) for c in feats])
    nuniq = len(pd.unique(y_raw.dropna()))
    task = ("binclass" if nuniq == 2 else
            "regression" if (pt.is_float_dtype(y_raw) and nuniq > 20) else "multiclass")
    if task != "regression":
        cls = {c: i for i, c in enumerate(sorted(pd.unique(y_raw.dropna())))}
        y = np.array([cls.get(v, 0) for v in y_raw.to_numpy()])
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype="float64")
    return X, y, tnum, task, len(feats)


def _load_tabred(ds, cfg):
    from src.data.tabred_loader import load_tabred
    from src.analysis.drift_measure import _stack
    data = load_tabred(ds, Path(cfg.data.root), split=cfg.experiment.split)
    # use the TRAIN portion (carries the within-train temporal axis); t = its timestamp
    X = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
    return X, data.train.y, data.train.t, data.task, X.shape[1]


# ----------------------------------------------------------------------------- synthetic ground truth
def _synth(kind, seed=0, n=12000, d=10):
    """Three regimes the instrument MUST separate, over a continuous time axis (-> quantile windows):
       concept   : rule P(y|x) rotates with t, covariates stationary  -> DEPLOYMENT-CONCEPT
       covariate : rule FIXED, covariate mean drifts with t           -> DEPLOYMENT-DECAY-COVARIATE
       stable    : rule fixed, covariates stationary                  -> DEPLOYMENT-STABLE
    """
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n))
    X = rng.normal(0, 1, (n, d))
    if kind == "concept":
        ang = 3.0 * t                                   # decision rule rotates over time
        score = np.cos(ang) * X[:, 0] + np.sin(ang) * X[:, 1]
        y = (3 * score + rng.normal(0, .4, n) > 0).astype(int)
    elif kind == "covariate":
        # P(x) drifts (X0 mean ramps out of the early support) but the rule in OBSERVED-feature
        # space is FIXED and NONLINEAR -> the old model must EXTRAPOLATE on the new region
        # (decay + recency_gain), yet old samples carry CORRECT labels -> staleness_harm <= 0.
        X[:, 0] = X[:, 0] + 6.0 * t
        y = (np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1] + rng.normal(0, .3, n) > 0).astype(int)
    else:                                               # stable
        y = (3 * X[:, 0] + rng.normal(0, .4, n) > 0).astype(int)
    return X, y, t, "binclass"


# ----------------------------------------------------------------------------- main
def _show(r):
    f = lambda x: f"{x:+.3f}" if isinstance(x, (int, float)) else "   -"
    ci = lambda c: f"[{f(c[0])},{f(c[1])}]" if c and c[0] is not None else "   -"
    print(f"  {r['dataset'][:20]:20s} W={r['n_windows']:>2d} "
          f"decay={f(r['decay'])} rec={f(r['recency_gain'])}{ci(r['recency_gain_ci'])} "
          f"stale={f(r['staleness_harm'])}{ci(r['staleness_harm_ci'])} => {r['verdict']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--target", default=None); ap.add_argument("--time", default=None)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--name", default=None)
    ap.add_argument("--tabred", nargs="*", default=[])
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--by-value", action="store_true",
                    help="one window per unique time value (e.g. EMBER YYYYMM months)")
    ap.add_argument("--windows", type=int, default=10, help="quantile windows when not --by-value")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--max-n", type=int, default=60000)
    ap.add_argument("--synth", action="store_true")
    args = ap.parse_args()
    out_dir = Path("results/phase1/deployment_decay"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    if args.synth:
        print("\n==== SYNTH ground-truth controls (instrument MUST land in the 3 regimes) ====")
        print("  EXPECT  concept=>DEPLOYMENT-CONCEPT  covariate=>DEPLOYMENT-DECAY-COVARIATE  stable=>DEPLOYMENT-STABLE")
        for kind in ("concept", "covariate", "stable"):
            X, y, t, task = _synth(kind)
            r = assess(f"synth_{kind}", X, y, t, task, K=args.windows, n_seeds=args.n_seeds,
                       max_train=args.max_train)
            rows.append(r); _show(r)
        (out_dir / "synth_summary.json").write_text(json.dumps({"rows": rows}, indent=2, default=float))
        ok = ({r["dataset"]: r["verdict"] for r in rows} ==
              {"synth_concept": "DEPLOYMENT-CONCEPT",
               "synth_covariate": "DEPLOYMENT-DECAY-COVARIATE",
               "synth_stable": "DEPLOYMENT-STABLE"})
        print(f"\n  GROUND-TRUTH {'PASS' if ok else 'CHECK (verdicts above must match EXPECT)'}")
        print(f"  wrote {out_dir}/synth_summary.json")
        return

    jobs = []
    if args.csv and args.target and args.time:
        X, y, t, task, nf = _load_csv(args.csv, args.target, args.time, args.drop, args.max_n)
        jobs.append((args.name or Path(args.csv).stem, X, y, t, task, nf))
    if args.tabred:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
        for ds in args.tabred:
            X, y, t, task, nf = _load_tabred(ds, cfg)
            jobs.append((ds, X, y, t, task, nf))
    if not jobs:
        print("provide --csv --target --time, or --tabred ..., or --synth"); return

    print("\n==== DEPLOYMENT-DECAY probe (rolling-origin: train past -> predict future) ====")
    for name, X, y, t, task, nf in jobs:
        print(f"  [{name}] task={task} feats={nf} n={len(y)}")
        r = assess(name, X, y, t, task, K=args.windows, by_value=args.by_value,
                   n_seeds=args.n_seeds, max_train=args.max_train)
        rows.append(r); _show(r)
    print("\n  PRE-REG: DEPLOYMENT-CONCEPT => within-overlap negative is an INSTRUMENT ARTIFACT (real")
    print("  positive, overlap lens was blind here). DEPLOYMENT-DECAY-COVARIATE => exploitable but")
    print("  covariate-mechanism. DEPLOYMENT-STABLE on all => negative hardened on both lenses => write it.")
    blob = json.loads((out_dir / "summary.json").read_text()) if (out_dir / "summary.json").exists() else {"rows": []}
    blob.setdefault("rows", []).extend(rows)
    (out_dir / "summary.json").write_text(json.dumps(blob, indent=2, default=float))
    print(f"\n  appended to {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
