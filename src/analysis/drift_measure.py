"""G3: measure how much exploitable temporal drift each dataset actually has.

Model-light, objective (sklearn only, no TabM/GPU). Answers the foundational
question the Phase-1 null hinges on: *is there drift to capture at all?*

  covariate_shift_auc(X_past, X_future): can a classifier tell past from future
      samples by their FEATURES? AUC>>0.5 => covariate (P(x)) shift. We measure
      train-vs-test (the temporal split's shift) and early-vs-late within train.
  label_drift(y, t): how does the target (pos-rate / mean) move over time bins?
      Spearman(t, y) + range across deciles => label/prior shift.

These are independent of our model, so they tell us whether Test-1/Test-2 nulls
are "no signal in the data" vs "our method missed it".

Uses HistGradientBoostingClassifier (handles NaN natively -> raw features ok).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from scipy.stats import spearmanr


def _stack(*arrs) -> Optional[np.ndarray]:
    cols = [np.asarray(a, dtype=np.float64) for a in arrs if a is not None and a.size]
    if not cols:
        return None
    return np.concatenate(cols, axis=1)


def _hgb_auc(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    clf.fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))


def _single_feature_aucs(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-column train/test separability using the feature value as the score."""
    aucs = np.full(X.shape[1], 0.5)
    for j in range(X.shape[1]):
        col = X[:, j].astype(np.float64)
        if np.all(np.isnan(col)):
            continue
        med = np.nanmedian(col)
        col = np.where(np.isnan(col), med, col)
        if np.std(col) == 0:
            continue
        a = roc_auc_score(y, col)
        aucs[j] = max(a, 1.0 - a)         # direction-agnostic
    return aucs


def covariate_shift_auc(
    X_past: np.ndarray, X_future: np.ndarray, *,
    seed: int = 0, max_n: int = 20_000, drop_top: int = 5,
) -> dict:
    """Past-vs-future separability by features, with trivial-vs-pervasive diagnostics.

    auc                 : multivariate HGB AUC (0.5=no shift, 1.0=perfectly separable)
    auc_drop_top{N}     : AUC after removing the N most time-separating features
                          (if it collapses to ~0.5 => shift was a few time-proxy cols;
                           if it stays high => pervasive multivariate drift)
    max_single_feat_auc : best single-feature separability
    n_feat_auc_gt_0.9   : how many features individually separate past/future
    """
    rng = np.random.default_rng(seed)

    def sub(X):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] > max_n:
            X = X[rng.choice(X.shape[0], max_n, replace=False)]
        return X

    Xa, Xb = sub(X_past), sub(X_future)
    X = np.concatenate([Xa, Xb], axis=0)
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])

    # Drop degenerate columns: all-NaN or constant break HGB's bin mapper.
    with np.errstate(all="ignore"):
        keep = (~np.all(np.isnan(X), axis=0)) & (np.nanstd(X, axis=0) > 0)
    if not keep.any():
        return {"auc": 0.5, "n_past": int(len(Xa)), "n_future": int(len(Xb)),
                "note": "no usable columns"}
    X = X[:, keep]

    auc = _hgb_auc(X, y, seed)
    sf = _single_feature_aucs(X, y)
    order = np.argsort(-sf)
    keep2 = np.ones(X.shape[1], dtype=bool)
    keep2[order[:drop_top]] = False
    auc_drop = _hgb_auc(X[:, keep2], y, seed) if keep2.any() else 0.5

    return {
        "auc": auc,
        f"auc_drop_top{drop_top}": auc_drop,
        "max_single_feat_auc": float(sf.max()),
        "n_feat_auc_gt_0.9": int((sf > 0.9).sum()),
        "n_features_kept": int(X.shape[1]),
        "n_past": int(len(Xa)), "n_future": int(len(Xb)),
    }


def _fit_eval(Xtr, ytr, Xte, yte, task, seed) -> float:
    """Fit HGB on (Xtr,ytr), score on (Xte,yte). regression->RMSE, classif->AUC.

    Drops all-NaN/constant columns of the TRAINING matrix (small strata can make a
    column degenerate -> HGB bin mapper 'window shape' crash)."""
    Xtr = np.asarray(Xtr, dtype=np.float64); Xte = np.asarray(Xte, dtype=np.float64)
    with np.errstate(all="ignore"):
        keep = (~np.all(np.isnan(Xtr), axis=0)) & (np.nanstd(Xtr, axis=0) > 0)
    if keep.any():
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    if task == "regression":
        m = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
        m.fit(Xtr, ytr)
        return float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))
    m = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
    m.fit(Xtr, ytr)
    if task == "multiclass":                      # AUC[:,1] is binary-only -> use accuracy
        return float(accuracy_score(yte, m.predict(Xte)))
    return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))


def concept_drift_gap(
    X_early, y_early, X_late, y_late, X_future, y_future, task, *,
    seed: int = 0, max_n: int = 20_000,
) -> dict:
    """Concept drift = does the predictive rule P(y|x) change over time?

    Train one model on EARLY data and one on LATE data (equal size), evaluate BOTH
    on the same held-out FUTURE set. If the late-trained model does much better on
    the future, the rule moved (concept drift); if they tie, the early rule still
    holds (no concept drift -> covariate shift only -> a time-indexed memory can't
    help prediction).

    gap_concept (direction-normalized, >0 => concept drift / late is better):
      regression : rmse_early - rmse_late
      classif    : auc_late  - auc_early
    """
    rng = np.random.default_rng(seed)

    def clean_and_sub(X, n=None):
        X = np.asarray(X, dtype=np.float64)
        if n is not None and X.shape[0] > n:
            X = X[rng.choice(X.shape[0], n, replace=False)]
        return X

    n = min(len(y_early), len(y_late), max_n)
    ie = rng.choice(len(y_early), n, replace=False)
    il = rng.choice(len(y_late), n, replace=False)
    Xe, ye = np.asarray(X_early, float)[ie], np.asarray(y_early)[ie]
    Xl, yl = np.asarray(X_late, float)[il], np.asarray(y_late)[il]
    Xf, yf = np.asarray(X_future, float), np.asarray(y_future)

    # Keep only columns valid in BOTH training sets (we fit on each separately):
    # a column all-NaN/constant within early or late breaks HGB's bin mapper even
    # if it has values elsewhere. (Xf may still contain NaN -> HGB handles at predict.)
    def _good(X):
        with np.errstate(all="ignore"):
            return (~np.all(np.isnan(X), axis=0)) & (np.nanstd(X, axis=0) > 0)
    keep = _good(Xe) & _good(Xl)
    if not keep.any():
        return {"gap_concept": 0.0, "gap_rel": 0.0, "n_each": int(n),
                "metric": {"regression": "rmse", "multiclass": "accuracy"}.get(task, "auc"),
                "note": "no usable columns"}
    Xe, Xl, Xf = Xe[:, keep], Xl[:, keep], Xf[:, keep]

    s_early = _fit_eval(Xe, ye, Xf, yf, task, seed)
    s_late = _fit_eval(Xl, yl, Xf, yf, task, seed)
    if task == "regression":
        gap = s_early - s_late                      # >0 => late better => concept drift
        rel = gap / (s_late + 1e-12)
    else:
        gap = s_late - s_early
        rel = gap / (abs(s_early - 0.5) + 1e-12)    # vs early's skill above chance
    return {"score_early_on_future": s_early, "score_late_on_future": s_late,
            "gap_concept": float(gap), "gap_rel": float(rel), "n_each": int(n),
            "metric": "rmse" if task == "regression" else "auc"}


def overlap_feasibility(
    X_early, y_early, X_late, y_late, task, *,
    seed: int = 0, max_n: int = 20_000, bands=((0.1, 0.9), (0.2, 0.8)),
) -> dict:
    """Can covariate-adjusted concept be MEASURED? (F3, PLAN_RESCUE §A)

    Under AUC≈1.0 the early/late x supports barely overlap -> 'fix x, vary t' is
    ill-posed. We probe, on a held-out set:
      - overlap mass = P(late|x) ∈ band (with band sensitivity 0.1-0.9 vs 0.2-0.8)
      - IW ESS over early points (reweight early->late): (Σw)²/Σw²  (low = poor overlap)
      - LABEL support: per time-half, # of the minority class within the overlap region
        (covariate overlap is useless if a half has ~no minority events to fit P(y|x)).
    """
    rng = np.random.default_rng(seed)

    def sub(X, y, n):
        X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
        if len(y) > n:
            ii = rng.choice(len(y), n, replace=False); return X[ii], y[ii]
        return X, y

    n = min(len(y_early), len(y_late), max_n)
    Xe, ye = sub(X_early, y_early, n); Xl, yl = sub(X_late, y_late, n)
    with np.errstate(all="ignore"):
        keep = ((~np.all(np.isnan(Xe), axis=0)) & (np.nanstd(Xe, axis=0) > 0)
                & (~np.all(np.isnan(Xl), axis=0)) & (np.nanstd(Xl, axis=0) > 0))
    if not keep.any():
        return {"measurable": False, "note": "no usable columns"}
    Xe, Xl = Xe[:, keep], Xl[:, keep]
    X = np.concatenate([Xe, Xl]); half = np.concatenate([np.zeros(len(ye)), np.ones(len(yl))])
    yy = np.concatenate([ye, yl])
    Xtr, Xte, htr, hte, _, yte = train_test_split(
        X, half, yy, test_size=0.4, random_state=seed, stratify=half)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    clf.fit(Xtr, htr)
    p = clf.predict_proba(Xte)[:, 1]                       # P(late|x) on held-out

    out = {"bands": {}}
    is_cls = task in ("binclass", "multiclass")
    for lob, hib in bands:
        m = (p >= lob) & (p <= hib)
        oe, ol = m & (hte == 0), m & (hte == 1)
        def minority(mask):
            yy_ = yte[mask]
            if len(yy_) == 0:
                return 0
            return int(min((yy_ == c).sum() for c in np.unique(yy))) if is_cls else int(len(yy_))
        out["bands"][f"{lob}-{hib}"] = {
            "overlap_mass": float(m.mean()),
            "n_overlap_early": int(oe.sum()), "n_overlap_late": int(ol.sum()),
            "minority_events_early": minority(oe), "minority_events_late": minority(ol)}
    pe = np.clip(p[hte == 0], 1e-4, 1 - 1e-4)
    w = pe / (1 - pe)
    out["IW_ESS_early"] = float((w.sum() ** 2) / ((w ** 2).sum() + 1e-12))
    return out


def _transfer_gap(Xe, ye, Xl, yl, task, seed):
    """Transfer gap on a FIXED late test: early-trained vs late-trained, SAME test set.
    >0 => early rule fails to transfer = concept (covariate matched by caller)."""
    if min(len(ye), len(yl)) < 150:
        return None
    if task != "regression":   # need both classes in early-train, late-train, late-test
        if len(np.unique(ye)) < 2 or len(np.unique(yl)) < 2:
            return None
    Xl_tr, Xl_te, yl_tr, yl_te = train_test_split(Xl, yl, test_size=0.5, random_state=seed)
    if task != "regression" and (len(np.unique(yl_tr)) < 2 or len(np.unique(yl_te)) < 2):
        return None
    se = _fit_eval(Xe, ye, Xl_te, yl_te, task, seed)        # early-trained on late-test
    sl = _fit_eval(Xl_tr, yl_tr, Xl_te, yl_te, task, seed)  # late-trained on SAME late-test
    gap = (se - sl) if task == "regression" else (sl - se)
    return {"score_early": se, "score_late": sl, "gap": float(gap),
            "n_early": int(len(ye)), "n_late": int(len(yl))}


def concept_within_overlap(
    X_early, y_early, X_late, y_late, task, *,
    seed: int = 0, max_n: int = 20_000, band=(0.1, 0.9), min_per_half: int = 200,
    permute_time: bool = False,
) -> dict:
    """Covariate-MATCHED concept measurement, restricted to the common-support band.

    Fixes the global-IW heavy-tail artifact (ESS collapse) by selecting the overlap
    region with P(late|x)∈band and comparing P(y|x) early-vs-late WITHIN it (no global
    reweighting). Both train pools and the eval set sit in the same band → covariate
    is matched → the gap is concept, not covariate.

    gap_concept (>0 => rule moved within common support):
      regression: rmse(early→late_test) − rmse(late→late_test)
      classif:    auc(late→late_test)   − auc(early→late_test)
    """
    rng = np.random.default_rng(seed)

    def sub(X, y, n):
        X = np.asarray(X, float); y = np.asarray(y)
        if len(y) > n:
            ii = rng.choice(len(y), n, replace=False); return X[ii], y[ii]
        return X, y

    n = min(len(y_early), len(y_late), max_n)
    Xe, ye = sub(X_early, y_early, n); Xl, yl = sub(X_late, y_late, n)
    with np.errstate(all="ignore"):
        keep = ((~np.all(np.isnan(Xe), axis=0)) & (np.nanstd(Xe, axis=0) > 0)
                & (~np.all(np.isnan(Xl), axis=0)) & (np.nanstd(Xl, axis=0) > 0))
    if not keep.any():
        return {"measurable": False, "note": "no usable columns"}
    Xe, Xl = Xe[:, keep], Xl[:, keep]
    X = np.concatenate([Xe, Xl]); half = np.concatenate([np.zeros(len(ye)), np.ones(len(yl))])
    yy = np.concatenate([ye, yl])
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    # OUT-OF-FOLD p for region selection (removes in-sample optimism in choosing overlap)
    p = cross_val_predict(clf, X, half, cv=5, method="predict_proba")[:, 1]
    ov = (p >= band[0]) & (p <= band[1])
    if permute_time:
        # PLACEBO (PLAN_V3 G1): permute the early/late label AMONG the overlap points,
        # keeping the same region/x-distribution. Breaks any real early→late structure,
        # so a non-zero gap here is the HOME-FIELD / N-asymmetry bias floor, not concept.
        idx_ov = np.where(ov)[0]
        half = half.copy(); half[idx_ov] = rng.permutation(half[idx_ov])
    eo, lo = ov & (half == 0), ov & (half == 1)
    n_e, n_l = int(eo.sum()), int(lo.sum())
    if min(n_e, n_l) < min_per_half:
        return {"measurable": False, "n_overlap_early": n_e, "n_overlap_late": n_l,
                "note": "too few overlap points per half"}
    full = _transfer_gap(X[eo], yy[eo], X[lo], yy[lo], task, seed)

    # p-stratified stability: if the gap were residual covariate, it would vary across
    # P(late|x) strata. Compute the transfer gap within p-tertiles of the overlap band.
    edges = np.quantile(p[ov], [0.0, 1 / 3, 2 / 3, 1.0])
    strata = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p <= b)
        g = _transfer_gap(X[m & (half == 0)], yy[m & (half == 0)],
                          X[m & (half == 1)], yy[m & (half == 1)], task, seed)
        strata.append(None if g is None else g["gap"])
    sg = [g for g in strata if g is not None]
    return {"measurable": True, "n_overlap_early": n_e, "n_overlap_late": n_l,
            "metric": {"regression": "rmse", "multiclass": "accuracy"}.get(task, "auc"),
            "score_early_on_lateOverlap": full["score_early"],
            "score_late_on_lateOverlap": full["score_late"],
            "concept_gap_within_overlap": full["gap"],
            "strata_gaps": strata,
            "strata_gap_min": (float(min(sg)) if sg else None),
            "strata_gap_max": (float(max(sg)) if sg else None),
            "note": "transfer gap on a FIXED late-overlap test (concept, not difficulty); "
                    "out-of-fold p; stable across p-strata => not residual covariate"}


def disde_iw_degeneration(
    X_early, X_late, *, seed: int = 0, max_n: int = 20_000, clip=1e-3,
) -> dict:
    """DISDE-style importance-reweighting DEGENERATION diagnostic (R2.2).

    DISDE (Cai, Namkoong, Yadlowsky, Operations Research 2025) attributes a
    performance drop to within-support Y|X change vs X-shift-into-unseen by
    reweighting the source (early) covariates to the target (late) distribution
    with density ratios w(x)=p_late(x)/p_early(x). Under STRONG covariate shift
    (the AUC≈1.0 regime that defines TabReD-style temporal drift), w has a heavy
    tail: a few near-out-of-support early points carry almost all the mass, so any
    self-normalized IW estimate Ê = Σw·ℓ / Σw has effective sample size
    ESS=(Σw)²/Σw² that collapses — the DISDE within-support term becomes
    unestimable. This function quantifies that degeneration WITHOUT fitting a
    probe model (ESS bounds the variance of ANY bounded reweighted functional).

    Returns (over early points; w from an OUT-OF-FOLD P(late|x) to avoid optimism):
      ess, ess_pct (=ESS / n_early·100), cv (=std(w)/mean(w)), max_weight_share
      (=max(w)/Σw, single-point leverage), n_early. Low ess_pct / high cv / high
      max_weight_share => DISDE reweighting is degenerate (use the within-overlap
      frame instead — concept_within_overlap, which restricts to the band, no
      global reweight).
    """
    rng = np.random.default_rng(seed)

    def sub(X, n):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] > n:
            X = X[rng.choice(X.shape[0], n, replace=False)]
        return X

    Xe, Xl = sub(X_early, max_n), sub(X_late, max_n)
    with np.errstate(all="ignore"):
        keep = ((~np.all(np.isnan(Xe), axis=0)) & (np.nanstd(Xe, axis=0) > 0)
                & (~np.all(np.isnan(Xl), axis=0)) & (np.nanstd(Xl, axis=0) > 0))
    if not keep.any():
        return {"note": "no usable columns"}
    Xe, Xl = Xe[:, keep], Xl[:, keep]
    X = np.concatenate([Xe, Xl])
    half = np.concatenate([np.zeros(len(Xe)), np.ones(len(Xl))])
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    p = cross_val_predict(clf, X, half, cv=5, method="predict_proba")[:, 1]
    # overlap_mass = support diagnostic (fraction of POOLED pts with P(late|x)∈[.1,.9]).
    # This is the PRIMARY degeneration signal: ~0 => early/late supports are DISJOINT,
    # so reweighting is biased (no early mass resembles late) — note ESS/CV are variance
    # diagnostics and MISS this bias mode (perfect separation clips all early w to the
    # floor => uniform tiny weights => ESS looks fine while the estimate is meaningless).
    overlap_mass = float(((p >= 0.1) & (p <= 0.9)).mean())
    pe = np.clip(p[half == 0], clip, 1 - clip)
    w = pe / (1 - pe)                                    # density ratio on early pts
    sw, sw2 = float(w.sum()), float((w ** 2).sum())
    ess = (sw ** 2) / (sw2 + 1e-12)
    return {
        "overlap_mass": overlap_mass,
        "ess": float(ess), "ess_pct": float(100.0 * ess / len(w)),
        "cv": float(w.std() / (w.mean() + 1e-12)),
        "max_weight_share": float(w.max() / (sw + 1e-12)),
        "n_early": int(len(w)),
    }


def label_drift(y: np.ndarray, t: np.ndarray, task: str, *, n_bins: int = 10) -> dict:
    """Target statistic per time decile + Spearman(t, y). (pos-rate / mean)."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    order = np.argsort(t, kind="stable")
    bins = np.array_split(order, n_bins)
    stat = []
    for b in bins:
        if len(b) == 0:
            continue
        stat.append(float(y[b].mean()))   # pos-rate (binclass) or mean (regression)
    stat = np.asarray(stat)
    rho, p = spearmanr(t, y)
    return {
        "per_bin": stat.tolist(),
        "range": float(stat.max() - stat.min()),
        "rel_range": float((stat.max() - stat.min()) / (abs(stat.mean()) + 1e-12)),
        "spearman_t_y": float(rho),
        "spearman_p": float(p),
        "task": task,
    }
