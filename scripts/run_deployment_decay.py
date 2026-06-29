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
_FIT_WARNED = set()
_DEBUG_RAISE = False     # set by --debug-raise: re-raise HGB errors with full traceback


def _fit_score(Xtr, ytr, Xte, yte, task, seed):
    """Higher-is-better score of a fresh HGB trained on (Xtr,ytr), eval on (Xte,yte).
    Returns None if the window can't yield a valid score (single-class train/test, fit error).
    HGB handles NaN natively; imbalance is preserved (no resampling). Any fit/predict error is
    swallowed to None (one-time warning) so a single degenerate window can't kill the batch."""
    Xtr = np.asarray(Xtr, float); Xte = np.asarray(Xte, float)
    # Drop columns constant WITHIN this training subset (not just globally). sklearn>=1.9's
    # bin-mapper computes midpoints between consecutive distinct values via sliding_window_view(.,2),
    # which raises "window shape cannot be larger than input array shape" on a single-distinct-value
    # column. A within-subset constant carries no signal, so dropping it is lossless and removes the
    # crash. (Bootstrap/sparse windows make this common on e.g. sberbank's 392 features.)
    keep = _nonconstant_mask(Xtr)
    if not keep.any():
        return None
    if not keep.all():
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    try:
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
    except Exception as e:                                  # version-specific binning quirks, etc.
        if _DEBUG_RAISE:
            raise
        key = type(e).__name__
        if key not in _FIT_WARNED:
            _FIT_WARNED.add(key)
            print(f"    [warn] HGB fit/score skipped a window: {key}: {e} "
                  f"(Xtr={np.asarray(Xtr).shape})", file=sys.stderr)
        return None


def _nonconstant_mask(X):
    """Columns with >= 2 distinct non-NaN values. Computed WITHOUT triggering numpy's
    'Degrees of freedom <= 0' warning: nanstd is evaluated only on columns that have at least one
    non-NaN value (so N>=1, ddof=0 -> N>0), never on an all-NaN column (N=0, the warning's cause).
    The all-NaN columns are simply excluded up front. Result is identical, output is clean."""
    X = np.asarray(X, float)
    not_all_nan = ~np.all(np.isnan(X), axis=0)
    keep = np.zeros(X.shape[1], dtype=bool)
    if not_all_nan.any():
        with np.errstate(all="ignore"):
            keep[not_all_nan] = np.nanstd(X[:, not_all_nan], axis=0) > 0
    return keep


def _sanitize(X):
    """inf -> NaN (HGB tolerates NaN, not always inf across versions) and drop all-NaN / constant
    columns (a zero-variance or all-missing column breaks the parallel bin-mapper on some sklearn
    builds — the cause of the TabReD 'sliding_window_view' crash). Mirrors covariate_shift_auc."""
    X = np.asarray(X, float)
    X[~np.isfinite(X)] = np.nan
    keep = _nonconstant_mask(X)
    return X[:, keep] if keep.any() else X


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
    """Draw n training indices. When the pool is larger than n, subsample WITHOUT replacement
    (varies per seed). When the pool is <= n, BOOTSTRAP (with replacement) to size n so that
    seeds still differ -> the across-seed CI reflects genuine sampling uncertainty instead of
    collapsing to width 0 (which would make 'CI excludes 0' vacuously true)."""
    if len(idx) <= n:
        return rng.choice(idx, n, replace=True)
    return rng.choice(idx, n, replace=False)


# ----------------------------------------------------------------------------- core
def _per_seed(X, y, win, task, K, max_train, seed):
    """One seed of the rolling-origin analysis. Returns per-window-averaged
    (decay, recency_gain, staleness_harm) or Nones where not computable."""
    rng = np.random.default_rng(seed)
    by_w = [np.where(win == k)[0] for k in range(K)]
    # robust "old" anchor: the EARLIEST window with enough data (skips a sparse historical tail,
    # e.g. EMBER's pre-2017 months) instead of hard-failing on the literal oldest window.
    old_w = next((k for k in range(K) if _ok_train(y[by_w[k]], task)), None)
    if old_w is None:
        return None
    old_pool = by_w[old_w]
    # --- old-window model for the DECAY curve (its own held-out early baseline) ---
    w0 = rng.permutation(old_pool); cut = int(len(w0) * 0.7)
    tr0, ho0 = _sample(w0[:cut], max_train, rng), w0[cut:]
    base = (_fit_score(X[tr0], y[tr0], X[ho0], y[ho0], task, seed)
            if _ok_train(y[tr0], task) and len(ho0) >= 20 else None)

    back = range(max(old_w + 1, K // 2), K)            # future windows, strictly after the old anchor
    decays, recs, stales = [], [], []
    for j in back:
        te = by_w[j]
        if len(te) < 20 or (task != "regression" and len(np.unique(y[te])) < 2):
            continue
        # decay of the old model on this future window
        if base is not None and _ok_train(y[tr0], task):
            s = _fit_score(X[tr0], y[tr0], X[te], y[te], task, seed)
            if s is not None:
                decays.append(base - s)
        # recent / old models on the SAME future window, size-matched at N.
        # recency_gain = recent - old. staleness_harm = recent - (recent + old): the recent rows
        # are IDENTICAL in both, so only the N extra OLD rows differ -> no density confound.
        recent_pool = by_w[j - 1]
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
    X = _sanitize(X)
    y = np.asarray(y)
    t = np.asarray(t, float)
    # drop rows with a non-finite target (regression) or non-finite time
    good = np.isfinite(t)
    if task == "regression":
        good = good & np.isfinite(y.astype(float))
    X, y, t = X[good], y[good], t[good]
    win, Keff = _assign_windows(t, K, by_value)
    keep_w = win >= 0
    X, y, t, win = X[keep_w], y[keep_w], t[keep_w], win[keep_w]

    # ---- domain guards / trust diagnostics (so a degenerate time axis or pure serial
    #      correlation can't be read as a real deployment verdict) ----
    uniq_t = int(np.unique(t).size)
    cov_el = None                                          # covariate movement first->last window
    try:
        fw, lw = X[win == win.min()], X[win == win.max()]
        if len(fw) >= 20 and len(lw) >= 20:
            from src.analysis.drift_measure import covariate_shift_auc
            cov_el = covariate_shift_auc(fw, lw).get("auc")
    except Exception:
        cov_el = None
    ys = y[np.argsort(t, kind="stable")].astype(float)     # label autocorrelation in time order
    ylag = (float(np.corrcoef(ys[1:], ys[:-1])[0, 1])
            if ys.size > 2 and np.std(ys) > 0 else None)
    flags = []
    if uniq_t < Keff:
        flags.append("few-unique-times")                   # can't form the windows we claim
    if cov_el is not None and cov_el < 0.55:
        flags.append("no-covariate-movement")              # the 'time' axis barely moves -> verdict weak
    if task == "binclass" and ylag is not None and abs(ylag) > 0.5:
        flags.append("autocorr-risk")                      # serial labels (elec2) can fake decay/recency
    trust = "ok" if not flags else ";".join(flags)

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
    # A null measurement is NOT evidence of stability. STABLE may only be declared when the
    # deployment quantities were actually computed and came back flat — otherwise NO-DATA.
    measured = decay is not None and rec is not None and stale is not None and len(decay_s) >= 2
    if not measured:
        verdict = "NO-DATA"
    elif stale_concept and decay_present:
        verdict = "DEPLOYMENT-CONCEPT"
    elif rec_present and decay_present:
        verdict = "DEPLOYMENT-DECAY-COVARIATE"
    elif not decay_present and not rec_present:
        verdict = "DEPLOYMENT-STABLE"
    else:
        verdict = "INCONCLUSIVE"
    return {"dataset": name, "task": task, "n": int(len(y)), "n_windows": int(Keff),
            "n_seeds_ok": len([v for v in decay_s if v is not None]),
            "decay": decay, "decay_ci": decay_ci,
            "recency_gain": rec, "recency_gain_ci": rec_ci,
            "staleness_harm": stale, "staleness_harm_ci": stale_ci,
            "n_unique_t": uniq_t, "cov_auc_early_late": cov_el,
            "y_lag1_autocorr": ylag, "trust": trust,
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


def _load_stream(data):
    """elec2 / insects: the natural deployment timeline is the WHOLE stream, so concatenate
    train+val+test and order by t_raw (the global stream index each split carries). t_raw is the
    time axis for windowing (a clean monotone order)."""
    from src.analysis.drift_measure import _stack
    parts = [data.train, data.val, data.test]
    X = np.concatenate([_stack(p.X_num, p.X_bin, p.X_cat) for p in parts], axis=0)
    y = np.concatenate([p.y for p in parts])
    tr = np.concatenate([p.t_raw for p in parts]).astype(float)
    o = np.argsort(tr, kind="stable")
    return X[o], y[o], tr[o], data.task, X.shape[1]


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
    trust = r.get("trust", "ok")
    tnote = "" if trust == "ok" else f"  [TRUST: {trust}]"
    print(f"  {r['dataset'][:20]:20s} W={r['n_windows']:>2d} "
          f"decay={f(r['decay'])} rec={f(r['recency_gain'])}{ci(r['recency_gain_ci'])} "
          f"stale={f(r['staleness_harm'])}{ci(r['staleness_harm_ci'])} => {r['verdict']}{tnote}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--target", default=None); ap.add_argument("--time", default=None)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--name", default=None)
    ap.add_argument("--tabred", nargs="*", default=[])
    ap.add_argument("--elec2", action="store_true")
    ap.add_argument("--insects", action="store_true")
    ap.add_argument("--insects-variant", default="incremental_balanced")
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--by-value", action="store_true",
                    help="one window per unique time value (e.g. EMBER YYYYMM months)")
    ap.add_argument("--windows", type=int, default=10, help="quantile windows when not --by-value")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--max-train", type=int, default=6000)
    ap.add_argument("--max-n", type=int, default=60000)
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--debug-raise", action="store_true",
                    help="re-raise HGB fit errors with full traceback (diagnose the binning crash)")
    args = ap.parse_args()
    if args.debug_raise:
        globals()["_DEBUG_RAISE"] = True
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
    if args.elec2:
        from src.data.elec2_loader import load_elec2
        jobs.append(("elec2", *_load_stream(load_elec2(split="temporal", seed=0))))
    if args.insects:
        from src.data.insects_loader import load_insects
        jobs.append((f"insects_{args.insects_variant}",
                     *_load_stream(load_insects(variant=args.insects_variant,
                                                split="temporal", seed=0))))
    if not jobs:
        print("provide --csv --target --time, --tabred ..., --elec2/--insects, or --synth"); return

    print("\n==== DEPLOYMENT-DECAY probe (rolling-origin: train past -> predict future) ====")
    for name, X, y, t, task, nf in jobs:
        print(f"  [{name}] task={task} feats={nf} n={len(y)}")
        try:
            r = assess(name, X, y, t, task, K=args.windows, by_value=args.by_value,
                       n_seeds=args.n_seeds, max_train=args.max_train)
        except Exception as e:                              # never let one dataset kill the batch
            import traceback; traceback.print_exc()
            r = {"dataset": name, "task": task, "verdict": "ERROR", "error": f"{type(e).__name__}: {e}",
                 "decay": None, "decay_ci": [None, None], "recency_gain": None,
                 "recency_gain_ci": [None, None], "staleness_harm": None,
                 "staleness_harm_ci": [None, None], "n_windows": 0}
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
