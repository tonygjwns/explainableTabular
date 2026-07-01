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
DSTAR = 0.96          # identifiability gate: D_strip >= DSTAR => old data can't reach the future
                      # covariate region, so a staleness null is UNIDENTIFIABLE (not "no concept").
                      # Derived from the power curve (staleness alive to AUC~0.985, dead by ~0.999).
INJ_STRENGTH = 2.5    # reference concept strength for the injection control (clears the floor in
                      # overlapping geometry — validated: nuisance_proxy staleness ~+0.2 at 2.5).
ROW_FLOOR = 200       # a window/old-anchor needs >= this many rows to be usable (NEW-4: a sparse
                      # early window shrinks N and biases staleness toward the null).
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
    # early_stopping=False (NOT the default 'auto', which turns ON only when n_samples>10000):
    # the staleness comparison trains on N and 2N, and at the real regime (N~6000) only the 2N arm
    # would cross the 'auto' threshold -> asymmetric regularization that biases the comparison
    # independent of any rule change (red-team Flaw 1). Fixed regularization for every fit removes it.
    HGB = dict(max_iter=200, early_stopping=False, random_state=seed)
    try:
        if task == "regression":
            m = HistGradientBoostingRegressor(**HGB).fit(Xtr, ytr)
            return -float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))   # -RMSE: higher better
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return None
        m = HistGradientBoostingClassifier(**HGB).fit(Xtr, ytr)
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


def _proxy_mask(X, y, t, task, max_n=20000, seed=0):
    """Return a KEEP mask (True = non-proxy). A time-PROXY feature is one that, ALONE, strongly
    separates early from late (single-feature early/late AUC > 0.70) AND has ~no task-predictive
    value (|rank-corr with y| < 0.05). Such a feature (a clock / row-id / drifting nuisance) does
    two bad things, both validated on synth: it inflates the disjointness measure D toward 1.0, and
    — critically — it BLINDS the staleness model (a drifting nuisance made staleness read +0.0004;
    stripping it recovered the true +0.236). A time-separating feature that IS predictive is REAL
    covariate drift and is KEPT. Predictive value is MUTUAL INFORMATION, not rank-correlation — a
    non-monotonic predictive feature (a sin rule) has ~0 rank-corr but high MI and must NOT be stripped."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    X = np.asarray(X, float); n, d = X.shape
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, max_n, replace=False) if n > max_n else np.arange(n)
    Xs = np.nan_to_num(X[idx], nan=0.0, posinf=0.0, neginf=0.0)
    ts_ = np.asarray(t, float)[idx]; yv = np.asarray(y)[idx]
    late = (ts_ > np.median(ts_)).astype(int)
    keep = np.ones(d, bool)
    if len(np.unique(late)) < 2:
        return keep
    tsep = np.full(d, 0.5)                                             # single-feat time separation
    for j in range(d):
        col = Xs[:, j]
        if np.std(col) == 0:
            continue
        a = roc_auc_score(late, col); tsep[j] = max(a, 1 - a)
    try:                                                              # single-feat predictive value (MI)
        mi = (mutual_info_regression(Xs, yv.astype(float), random_state=seed) if task == "regression"
              else mutual_info_classif(Xs, yv.astype(int), random_state=seed))
    except Exception:
        return keep                                                  # MI failed -> strip nothing (safe)
    mi_cut = max(0.01, 0.05 * float(np.max(mi))) if mi.size else 0.01  # < 5% of the top feature's MI
    keep[(tsep > 0.70) & (mi < mi_cut)] = False
    return keep


def _disjointness(X, win, Keff, seed=0):
    """D = median over back-half windows W_j of held-out AUC(oldest-window vs W_j). Computed on the
    (proxy-stripped) feature space -> 'can old data even reach the future covariate region?'."""
    from sklearn.model_selection import train_test_split
    by_w = [np.where(win == k)[0] for k in range(Keff)]
    old = by_w[0]
    if len(old) < 40:
        return None
    aucs = []
    for j in range(max(1, Keff // 2), Keff):
        fut = by_w[j]
        if len(fut) < 40:
            continue
        nn = min(len(old), len(fut), 5000)
        Xd = np.nan_to_num(np.vstack([X[old[:nn]], X[fut[:nn]]]))
        yd = np.r_[np.zeros(nn), np.ones(nn)]
        try:
            a, b, c, dd = train_test_split(Xd, yd, test_size=0.3, random_state=seed, stratify=yd)
            m = HistGradientBoostingClassifier(max_iter=100, early_stopping=False,
                                               random_state=seed).fit(a, c)
            aucs.append(roc_auc_score(dd, m.predict_proba(b)[:, 1]))
        except Exception:
            continue
    return float(np.median(aucs)) if aucs else None


def _z(v):
    v = np.asarray(v, float); s = np.std(v)
    return (v - np.mean(v)) / (s + 1e-12)


def _inject_concept(X, t, task, strength=2.5, seed=0):
    """Replace y with a KNOWN time-ROTATING rule of controlled strength on the 2 highest-variance
    features, keeping this dataset's REAL covariate geometry (X, t). Probe for the injection control:
    run staleness on (X, y_injected). If it stays null on a candidate-UNIDENTIFIABLE dataset, the
    blindness is DEMONSTRATED on real geometry (UNIDENTIFIABLE-EARNED); if it recovers, D was
    misleading and the real null was informative (the dataset was identifiable after all)."""
    rng = np.random.default_rng(seed)
    Xf = np.nan_to_num(np.asarray(X, float))
    tn = (t - np.min(t)) / (np.max(t) - np.min(t) + 1e-12)
    order = np.argsort(-Xf.std(0))
    f0, f1 = (order[0], order[1]) if Xf.shape[1] >= 2 else (0, 0)
    ang = strength * tn
    score = np.cos(ang) * _z(Xf[:, f0]) + np.sin(ang) * _z(Xf[:, f1]) + rng.normal(0, .3, len(tn))
    if task == "regression":
        return score.astype(float)
    if task == "multiclass":
        s2 = np.cos(ang + 2.0) * _z(Xf[:, f0]) + np.sin(ang + 2.0) * _z(Xf[:, f1])
        return np.stack([score, s2, -score - s2], axis=1).argmax(1)
    return (score > np.median(score)).astype(int)


def _injection_recovers(X, t, task, K, by_value, max_train, n_seeds=4):
    """Inject a reference-strength concept into (X, t) and test whether staleness fires. Returns
    (recovered_bool, injected_staleness_mean)."""
    y_inj = _inject_concept(X, t, task, strength=INJ_STRENGTH)
    st = []
    for s in range(n_seeds):
        out = _per_seed(X, y_inj, t, task, K, by_value, max_train, s)
        if out is not None and out[2] is not None:
            st.append(out[2])
    m, ci = _ci95(st)
    recovered = m is not None and ci[0] is not None and ci[0] > 0 and m > FLOOR_GAIN
    return recovered, m


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
def _assign_windows(t, K, by_value, rng=None):
    """Return per-row window id in [0..K-1] (or fewer). by_value: one window per unique time
    value (e.g. EMBER YYYYMM months = TESSERACT-faithful). else: K bins by TIMESTAMP-VALUE rank so
    all rows sharing a timestamp land in the SAME window (ties never straddle the past/future
    boundary — red-team Flaw 5). When rng is given, the K-1 window BOUNDARIES are jittered by a few
    unique-value ranks (block-bootstrap over timestamp values): this varies the partition across
    seeds even on COARSE timestamps, where a 90% row subsample leaves np.unique(t) unchanged and
    would otherwise FREEZE the partition -> under-dispersed CI (red-team NEW-3, reproduced)."""
    t = np.asarray(t, float)
    if by_value:
        uniq = np.unique(t[np.isfinite(t)])
        remap = {v: i for i, v in enumerate(uniq)}
        return np.array([remap.get(v, -1) for v in t], int), len(uniq)
    uniq, inv = np.unique(t, return_inverse=True)      # inv = rank of each row's timestamp value
    U = len(uniq)
    if rng is not None and U > K:
        edges = (np.arange(1, K) * U / K)              # K-1 interior boundaries in value-rank space
        # jitter = a FEW unique timestamps (capped 1..3): meaningful when timestamps are chunky
        # (coarse: ±1 value moves a whole block -> partition varies) and negligible when fine
        # (±few rows). Scaling with U over-jittered fine datasets (±25% of a window).
        b = int(np.clip(U // (4 * K), 1, 3))
        edges = np.clip(np.sort(edges + rng.integers(-b, b + 1, size=K - 1)), 1, U - 1)
        w = np.searchsorted(edges, inv, side="right")
    else:
        w = (inv * K // U).clip(0, K - 1)
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
def _per_seed(X, y, t, task, K, by_value, max_train, seed):
    """One seed. A per-seed row SUBSAMPLE (90%) is windowed fresh, so the partition itself varies
    across seeds -> the across-seed CI captures partition + data-draw uncertainty, not only the
    training bootstrap (red-team Flaw 3). Returns (decay, recency_gain, staleness_harm, Keff).

    staleness_harm = score(recent N) − score(recent N ∪ old N): does ADDING old data on top of the
    SAME recent set hurt the future? >0 = old data is net-harmful = old labels contradict the current
    rule = concept. Under a fixed rule old data (correct labels) only adds coverage -> stale <= 0.
    NOTE on the red-team's proposed 'fixed-size' variant (recent∪old vs recent∪recent', both 2N): we
    implemented and ground-truth-tested it and REJECTED it — it leaks a covariate-coverage confound
    (recent' covers the future region better than old, so it reads stale>0 under a FIXED rule; the
    covariate synth went +0.000 -> +0.005). The N-vs-2N size effect it was meant to remove is
    empirically negligible (a d=10..700 bias sweep: N-vs-2N ≈ fixed-size on concept data), and
    early_stopping=False removes the only sharp asymmetry, so the original design is kept."""
    rng = np.random.default_rng(seed)
    n = len(y)
    sub = rng.choice(n, int(0.9 * n), replace=False) if n > 200 else np.arange(n)
    Xs, ys, ts = X[sub], y[sub], t[sub]
    win, Keff = _assign_windows(ts, K, by_value, rng=rng)   # rng => per-seed boundary jitter (NEW-3)
    keep = win >= 0
    Xs, ys, win = Xs[keep], ys[keep], win[keep]
    by_w = [np.where(win == k)[0] for k in range(Keff)]
    # robust "old" anchor: the EARLIEST window with enough ROWS (>=ROW_FLOOR, NEW-4) AND a valid
    # training set — skips a sparse historical tail whose tiny N would bias staleness toward null.
    old_w = next((k for k in range(Keff)
                  if len(by_w[k]) >= ROW_FLOOR and _ok_train(ys[by_w[k]], task)), None)
    if old_w is None:
        return None
    old_pool = by_w[old_w]
    # --- old-window model for the DECAY curve (its own held-out early baseline) ---
    w0 = rng.permutation(old_pool); cut = int(len(w0) * 0.7)
    tr0, ho0 = _sample(w0[:cut], max_train, rng), w0[cut:]
    base = (_fit_score(Xs[tr0], ys[tr0], Xs[ho0], ys[ho0], task, seed)
            if _ok_train(ys[tr0], task) and len(ho0) >= 20 else None)

    decays, recs, stales = [], [], []
    for j in range(max(old_w + 1, Keff // 2), Keff):       # future windows, after the old anchor
        te = by_w[j]
        if len(te) < 20 or (task != "regression" and len(np.unique(ys[te])) < 2):
            continue
        if base is not None and _ok_train(ys[tr0], task):
            s = _fit_score(Xs[tr0], ys[tr0], Xs[te], ys[te], task, seed)
            if s is not None:
                decays.append(base - s)
        recent_pool = by_w[j - 1]
        N = min(len(recent_pool), len(old_pool), max_train)
        if N < 50:
            continue
        recent = _sample(recent_pool, N, rng)
        old = _sample(old_pool, N, rng)
        if not (_ok_train(ys[recent], task) and _ok_train(ys[old], task)):
            continue
        recent_old = np.concatenate([recent, old])     # same N recent + N old
        sRec = _fit_score(Xs[recent], ys[recent], Xs[te], ys[te], task, seed)
        sOld = _fit_score(Xs[old], ys[old], Xs[te], ys[te], task, seed)
        sRecOld = _fit_score(Xs[recent_old], ys[recent_old], Xs[te], ys[te], task, seed)
        if None not in (sRec, sOld):
            recs.append(sRec - sOld)
        if None not in (sRec, sRecOld):
            stales.append(sRec - sRecOld)              # >0 = adding old HURT = concept
    mean = lambda a: float(np.mean(a)) if a else None
    return mean(decays), mean(recs), mean(stales), Keff


def _ci95(a):
    """Mean ± t(0.975, n-1)·SE. Uses the Student-t multiplier (n=5 -> 2.776, not 1.96) so a small
    seed count does not produce an anti-conservative interval (red-team Flaw 3)."""
    a = np.asarray([v for v in a if v is not None], float)
    if len(a) == 0:
        return None, [None, None]
    if len(a) < 2:
        return float(a[0]), [float(a[0]), float(a[0])]
    from scipy.stats import t as _student_t
    m, se = float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))
    tm = float(_student_t.ppf(0.975, len(a) - 1))
    return m, [m - tm * se, m + tm * se]


def assess(name, X, y, t, task, K=10, by_value=False, n_seeds=5, max_train=6000):
    X = _sanitize(X)
    y = np.asarray(y)
    t = np.asarray(t, float)
    # drop rows with a non-finite target (regression) or non-finite time
    good = np.isfinite(t)
    if task == "regression":
        good = good & np.isfinite(y.astype(float))
    X, y, t = X[good], y[good], t[good]
    # Regression: standardize the target (z-score) so -RMSE is in std units. The CONCEPT/STABLE
    # floor (0.02) is then unit-invariant — without this a target rescaled x100 would flip the
    # verdict (red-team Flaw 4). For AUC/accuracy the scores are already in [0,1].
    if task == "regression":
        yf = np.asarray(y, float); mu, sd = float(np.mean(yf)), float(np.std(yf)) + 1e-12
        y = (yf - mu) / sd

    # PROXY-STRIP (load-bearing, validated on synth): remove time-proxy features (single-feature
    # early/late AUC>0.70 AND ~no predictive value) from the MODEL space AND from D. A drifting
    # nuisance feature both inflates D->1 and BLINDS the staleness model (nuisance made staleness
    # read +0.0004; stripping recovered +0.236). Real (predictive) covariate drift is kept.
    keep_px = _proxy_mask(X, y, t, task)
    n_proxy = int((~keep_px).sum())
    X_full = X
    if not keep_px.all() and keep_px.any():
        X = X[:, keep_px]

    win, Keff = _assign_windows(t, K, by_value)            # single windowing, for diagnostics + D
    keep_w = win >= 0
    Xd, yd, td, win = X[keep_w], y[keep_w], t[keep_w], win[keep_w]
    # disjointness on stripped (D_strip, used for the identifiability gate) and full (D_full) space:
    # D_full>=D* but D_strip<D* => PROXY-SENSITIVE (the disjointness was a clock, not covariate).
    D_strip = _disjointness(Xd, win, Keff)
    D_full = _disjointness(X_full[keep_w], win, Keff)
    # time-shuffle control: shuffle t -> windows become random in time. Genuine covariate DRIFT
    # collapses to D~0.5; if D stays high the windows are separated by a STATIC feature (an id/index
    # the strip missed), not drift -> D-GATE-SUSPECT.
    D_shuffle = None
    if D_strip is not None and D_strip >= DSTAR:
        sh = np.random.default_rng(0).permutation(len(td))
        wsh, Ksh = _assign_windows(td[sh], Keff, False)
        D_shuffle = _disjointness(Xd, wsh, Ksh)
    min_window_n = int(min((int((win == k).sum()) for k in range(Keff) if (win == k).any()),
                           default=0))                        # NEW-4: smallest window row count

    # ---- domain guards / trust diagnostics (so a degenerate time axis or pure serial
    #      correlation can't be read as a real deployment verdict) ----
    uniq_t = int(np.unique(td).size)
    cov_el = None                                          # covariate movement first->last window
    try:
        fw, lw = Xd[win == win.min()], Xd[win == win.max()]
        if len(fw) >= 20 and len(lw) >= 20:
            from src.analysis.drift_measure import covariate_shift_auc
            cov_el = covariate_shift_auc(fw, lw).get("auc")
    except Exception:
        cov_el = None
    ys = yd[np.argsort(td, kind="stable")].astype(float)   # label autocorrelation in time order
    ylag = (float(np.corrcoef(ys[1:], ys[:-1])[0, 1])
            if ys.size > 2 and np.std(ys) > 0 else None)
    # tie-overlap: fraction of adjacent windows whose [min_t,max_t] overlap (a tied timestamp split
    # across the past/future boundary -> potential leakage the few-unique-times guard misses, Flaw 5)
    rngs = [(td[win == k].min(), td[win == k].max()) for k in range(Keff) if (win == k).any()]
    tie_ov = (sum(1 for a, b in zip(rngs, rngs[1:]) if a[1] >= b[0]) / max(len(rngs) - 1, 1)
              if len(rngs) > 1 else 0.0)

    decay_s, rec_s, stale_s = [], [], []
    for s in range(n_seeds):
        out = _per_seed(X, y, t, task, K, by_value, max_train, s)
        if out is None:
            continue
        d, r, st, Keff = out
        decay_s.append(d); rec_s.append(r); stale_s.append(st)
    decay, decay_ci = _ci95(decay_s)
    rec, rec_ci = _ci95(rec_s)
    stale, stale_ci = _ci95(stale_s)

    # ---- trust flags (informational; never override the verdict) ----
    flags = []
    if uniq_t < Keff:
        flags.append("few-unique-times")                   # can't form the windows we claim
    if cov_el is not None and cov_el < 0.55:
        flags.append("no-covariate-movement")              # the 'time' axis barely moves
    if task == "binclass" and ylag is not None and abs(ylag) > 0.5:
        flags.append("autocorr-risk")                      # serial labels (elec2) fake decay/recency
    if decay is not None and decay < -FLOOR_DECAY:
        flags.append("future-easier")                      # negative decay -> time axis may not be chronological
    if tie_ov > 0.3:
        flags.append("tie-split")                          # tied timestamps straddle window boundaries
    if n_proxy:
        flags.append(f"proxy-stripped:{n_proxy}")

    rec_present = (rec is not None and rec_ci[0] is not None
                   and rec_ci[0] > 0 and rec > FLOOR_GAIN)
    # v2 decision logic (3 converging reviewers): condition on staleness-CI-vs-floor AND the
    # identifiability gate D_strip vs D*. POSITIVE staleness ALWAYS overrides the gate (never file
    # a firing concept signal as unidentifiable). D is on the proxy-stripped space.
    stale_pos = (stale is not None and stale_ci[0] is not None
                 and stale_ci[0] > 0 and stale > FLOOR_GAIN)          # lower_CI > floor
    stale_sub = (stale is not None and stale_ci[0] is not None
                 and stale_ci[0] > 0 and not stale_pos)               # 0 < lower_CI <= floor
    stale_null = (stale is not None and stale_ci[1] is not None
                  and stale_ci[1] <= FLOOR_GAIN)                      # upper_CI <= floor
    disjoint = D_strip is not None and D_strip >= DSTAR
    if D_full is not None and D_strip is not None and D_full >= DSTAR and D_strip < DSTAR:
        flags.append("proxy-sensitive")                    # high disjointness was a clock, not covariate
    measured = decay is not None and rec is not None and stale is not None and len(decay_s) >= 2
    if not measured:
        verdict = "NO-DATA"
    elif stale_pos:                                         # concept fired — overrides D
        verdict = "DEPLOYMENT-CONCEPT"
        if disjoint:
            flags.append("d-gate-invalid")                 # fired where D declared it blind
    elif disjoint:                                          # staleness not positive AND support disjoint
        verdict = "UNIDENTIFIABLE-EXPLOITABLE" if rec_present else "UNIDENTIFIABLE-INERT"
    elif stale_sub:
        verdict = "SUBFLOOR-CONCEPT-SIGNAL"
    elif stale_null:                                        # identifiable region, no strong concept
        verdict = "DEPLOYMENT-DECAY-COVARIATE" if rec_present else "NO-STRONG-CONCEPT"
    else:
        verdict = "INCONCLUSIVE"

    # ---- injection control (breaks the circularity: is UNIDENTIFIABLE 'by assumption' or DEMONSTRATED?)
    inj_stale = None
    if verdict.startswith("UNIDENTIFIABLE"):
        recovered, inj_stale = _injection_recovers(X, t, task, K, by_value, max_train)
        if recovered:                                      # geometry HAD power -> the real null was informative
            verdict = "INJECTION-RECOVERED"; flags.append("injection-recovered")
        else:                                              # blindness demonstrated on real geometry
            flags.append("unident-earned")
        if D_shuffle is not None and D_shuffle > 0.6:
            flags.append("d-gate-suspect")                 # windows separated by a STATIC feature, not drift
    if min_window_n and min_window_n < ROW_FLOOR:
        flags.append(f"sparse-window:{min_window_n}")      # NEW-4
    trust = "ok" if not flags else ";".join(flags)
    return {"dataset": name, "task": task, "n": int(len(y)), "n_windows": int(Keff),
            "n_seeds_ok": len([v for v in decay_s if v is not None]),
            "decay": decay, "decay_ci": decay_ci,
            "recency_gain": rec, "recency_gain_ci": rec_ci,
            "staleness_harm": stale, "staleness_harm_ci": stale_ci,
            "D_strip": D_strip, "D_full": D_full, "D_shuffle": D_shuffle, "injected_staleness": inj_stale,
            "n_proxy_stripped": n_proxy, "min_window_n": min_window_n, "Dstar": DSTAR,
            "n_unique_t": uniq_t, "cov_auc_early_late": cov_el,
            "y_lag1_autocorr": ylag, "tie_overlap": float(tie_ov), "trust": trust,
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
    elif kind == "nuisance_proxy":
        # strong concept on X0,X1 (stationary) + a DRIFTING NUISANCE feature X2 (not in the rule).
        # Without proxy-strip the nuisance inflates D->1 and blinds the model -> wrongly reads
        # UNIDENTIFIABLE/no-concept; WITH strip it must read DEPLOYMENT-CONCEPT. Validates the strip.
        X[:, 2] = X[:, 2] + 8.0 * t
        ang = 2.5 * t
        y = (np.cos(ang) * X[:, 0] + np.sin(ang) * X[:, 1] + rng.normal(0, .3, n) > 0).astype(int)
    elif kind == "covariate_mc":
        # ADVERSARIAL control for the insects (multiclass / accuracy) read: P(x) drifts AND the
        # class PRIOR drifts with it, but P(y|x) is FIXED in observed-feature space. A true concept
        # discriminator must NOT call this CONCEPT — if staleness_harm fires here, the multiclass/
        # accuracy path is confounding prior-shift with a changed rule (the insects worry).
        X[:, 0] = X[:, 0] + 6.0 * t                     # covariate drift -> also shifts class prior
        logits = np.stack([np.sin(1.5 * X[:, 0]), 0.8 * X[:, 1], -0.7 * X[:, 2]], axis=1)
        y = (logits + rng.normal(0, .3, (n, 3))).argmax(1)        # FIXED 3-class rule
        return X, y, t, "multiclass"
    else:                                               # stable
        y = (3 * X[:, 0] + rng.normal(0, .4, n) > 0).astype(int)
    return X, y, t, "binclass"


# ----------------------------------------------------------------------------- main
def _show(r):
    f = lambda x: f"{x:+.3f}" if isinstance(x, (int, float)) else "   -"
    ci = lambda c: f"[{f(c[0])},{f(c[1])}]" if c and c[0] is not None else "   -"
    trust = r.get("trust", "ok")
    tnote = "" if trust == "ok" else f"  [TRUST: {trust}]"
    dstr = r.get("D_strip"); df = f"{dstr:.3f}" if isinstance(dstr, (int, float)) else "  -"
    print(f"  {r['dataset'][:20]:20s} W={r['n_windows']:>2d} D={df} "
          f"rec={f(r['recency_gain'])}{ci(r['recency_gain_ci'])} "
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
        print("  ADVERSARIAL  covariate_mc (multiclass prior-shift, FIXED rule) MUST NOT be CONCEPT")
        print("  ADVERSARIAL  nuisance_proxy (concept + drifting nuisance) MUST read CONCEPT (proxy-stripped)")
        for kind in ("concept", "covariate", "stable", "covariate_mc", "nuisance_proxy"):
            X, y, t, task = _synth(kind)
            r = assess(f"synth_{kind}", X, y, t, task, K=args.windows, n_seeds=args.n_seeds,
                       max_train=args.max_train)
            rows.append(r); _show(r)
        (out_dir / "synth_summary.json").write_text(json.dumps({"rows": rows}, indent=2, default=float))
        verdicts = {r["dataset"]: r["verdict"] for r in rows}
        C = lambda k: verdicts.get(f"synth_{k}", "")
        # v2 invariants: CONCEPT fires ONLY for true concept + recovered-nuisance; NEVER for
        # covariate/stable/prior-shift. And strong PREDICTIVE covariate drift must land UNIDENTIFIABLE
        # (the gate correctly abstains — it can't be told from concept-in-the-new-region).
        ok = (C("concept") == "DEPLOYMENT-CONCEPT"
              and C("nuisance_proxy") == "DEPLOYMENT-CONCEPT"           # proxy-strip recovers concept
              and C("covariate") != "DEPLOYMENT-CONCEPT"                # never FALSELY concept
              and C("stable") != "DEPLOYMENT-CONCEPT"
              and C("covariate_mc") != "DEPLOYMENT-CONCEPT")            # prior-shift adversarial
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
