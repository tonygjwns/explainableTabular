"""Shim model classes that mimic the HistGradientBoosting constructor signature
(max_iter, early_stopping, random_state, ...) but wrap other sklearn model classes.
Used to monkeypatch scripts/run_deployment_decay.py (dd module) for the model-class
dependence audit of the deployment-decay instrument.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def _build(kind, is_clf, seed):
    if kind == "linear":
        base = (LogisticRegression(max_iter=2000, random_state=seed) if is_clf
                else Ridge(random_state=seed))
        steps = [("imp", SimpleImputer(strategy="median")),
                 ("sc", StandardScaler()), ("m", base)]
    elif kind == "rf":
        base = (RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1) if is_clf
                else RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1))
        steps = [("imp", SimpleImputer(strategy="median")), ("m", base)]
    elif kind == "knn":
        base = (KNeighborsClassifier(n_neighbors=25) if is_clf
                else KNeighborsRegressor(n_neighbors=25))
        steps = [("imp", SimpleImputer(strategy="median")),
                 ("sc", StandardScaler()), ("m", base)]
    else:
        raise ValueError(kind)
    return Pipeline(steps)


class _ShimBase:
    _kind = None
    _is_clf = True

    def __init__(self, **kwargs):
        # accept and ignore HGB-specific kwargs (max_iter, early_stopping, ...)
        seed = kwargs.get("random_state", 0)
        if seed is None:
            seed = 0
        self._pipe = _build(self._kind, self._is_clf, int(seed))

    def fit(self, X, y):
        self._pipe.fit(np.asarray(X, float), y)
        return self

    def predict(self, X):
        return self._pipe.predict(np.asarray(X, float))

    def predict_proba(self, X):
        return self._pipe.predict_proba(np.asarray(X, float))

    @property
    def classes_(self):
        return self._pipe.classes_  # sklearn Pipeline delegates classes_ to final estimator


def make_shims(kind):
    """Return (ClassifierShim, RegressorShim) classes for the given kind."""
    clf = type(f"Shim_{kind}_clf", (_ShimBase,), {"_kind": kind, "_is_clf": True})
    reg = type(f"Shim_{kind}_reg", (_ShimBase,), {"_kind": kind, "_is_clf": False})
    return clf, reg


if __name__ == "__main__":
    # toy verification: classes_ delegation + predict_proba + HGB-style kwargs accepted
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 5)); X[0, 0] = np.nan
    y = (X[:, 1] > 0).astype(int)
    for kind in ("linear", "rf", "knn"):
        C, R = make_shims(kind)
        m = C(max_iter=200, early_stopping=False, random_state=3).fit(X, y)
        p = m.predict_proba(X)
        assert p.shape == (200, 2), p.shape
        assert list(m.classes_) == [0, 1], m.classes_
        r = R(max_iter=100, early_stopping=False, random_state=3).fit(X, y.astype(float))
        pr = r.predict(X)
        assert pr.shape == (200,)
        print(f"{kind}: OK classes_={list(m.classes_)} proba0={p[0].round(3).tolist()}")
    print("toy verification PASS")
