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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def _stack(*arrs) -> Optional[np.ndarray]:
    cols = [np.asarray(a, dtype=np.float64) for a in arrs if a is not None and a.size]
    if not cols:
        return None
    return np.concatenate(cols, axis=1)


def covariate_shift_auc(
    X_past: np.ndarray, X_future: np.ndarray, *,
    seed: int = 0, max_n: int = 20_000,
) -> dict:
    """AUC of a classifier separating past vs future by features (0.5=no shift)."""
    rng = np.random.default_rng(seed)

    def sub(X):
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] > max_n:
            X = X[rng.choice(X.shape[0], max_n, replace=False)]
        return X

    Xa, Xb = sub(X_past), sub(X_future)
    X = np.concatenate([Xa, Xb], axis=0)
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    clf.fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    return {"auc": auc, "n_past": int(len(Xa)), "n_future": int(len(Xb))}


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
