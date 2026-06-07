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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
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
    """Fit HGB on (Xtr,ytr), score on (Xte,yte). regression->RMSE, classif->AUC."""
    if task == "regression":
        m = HistGradientBoostingRegressor(max_iter=300, random_state=seed)
        m.fit(Xtr, ytr)
        return float(np.sqrt(mean_squared_error(yte, m.predict(Xte))))
    m = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
    m.fit(Xtr, ytr)
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

    # drop columns degenerate across the union (consistent feature set)
    U = np.concatenate([Xe, Xl, Xf], axis=0)
    with np.errstate(all="ignore"):
        keep = (~np.all(np.isnan(U), axis=0)) & (np.nanstd(U, axis=0) > 0)
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
