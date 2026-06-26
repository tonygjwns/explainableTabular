"""V3.5 breadth — a PANEL of river synthetic streams with KNOWN, DIAL-ABLE concept drift.

The generative test (run_correct_assumption) needs many datasets where the concept-drift
magnitude *varies* so that Spearman(concept_gap, recency_gain) is powered (the current n=1,
INSECTS-only, is the weakness). river's synth generators give ground-truth control: by
switching the labelling function at the stream midpoint (ConceptDriftStream) we create a
real P(y|x) change of a known size, and by NOT switching it we get a matched no-drift control
(concept ≈ 0). Across generator families (SEA, Agrawal, STAGGER, Sine, Hyperplane) and drift
shapes (none / abrupt / gradual / incremental) this yields ~12 streams spanning concept≈0 to
large — exactly the spread the cross-dataset correlation needs.

Each stream is returned as a TabReDDataset (task='binclass') so the within-overlap measure and
the generative test run unchanged. t = stream position (the drift trajectory); split='temporal'.

Requires `river` (already used by insects_loader): pip install river. Server-side.
"""
from __future__ import annotations

import numpy as np

from .tabred_loader import TabularSplit, TabReDDataset


# Each builder takes (seed, n) and returns a river stream iterator yielding (x_dict, y).
# Drift streams switch the labelling function at the midpoint => a real concept change.
def _concat(a, b, n):
    """Truly-abrupt concept switch: first n/2 from stream a, then from b (a HARD switch
    at the midpoint — same feature keys since a,b are the same generator family). Avoids
    river's ConceptDriftStream sigmoid, which OVERFLOWS at small width (exp(-4·Δ/width))."""
    def gen():
        h = n // 2
        for i, xy in enumerate(a):
            if i >= h:
                break
            yield xy
        for i, xy in enumerate(b):
            if i >= n - h:
                break
            yield xy
    return gen()


def _panel(n):
    from river.datasets import synth

    def grad(a, b, seed):                             # gradual blend A->B (wide sigmoid: safe)
        return synth.ConceptDriftStream(stream=a, drift_stream=b, seed=seed,
                                        position=n // 2, width=n // 5)
    P = {}
    # ---- SEA (4 variants = 4 thresholds) ----
    P["sea_nodrift"] = lambda s: synth.SEA(variant=0, seed=s)
    P["sea_abrupt"] = lambda s: _concat(synth.SEA(variant=0, seed=s), synth.SEA(variant=3, seed=s), n)
    P["sea_gradual"] = lambda s: grad(synth.SEA(variant=0, seed=s), synth.SEA(variant=3, seed=s), s)
    # ---- Agrawal (10 functions) ----
    P["agrawal_nodrift"] = lambda s: synth.Agrawal(classification_function=0, seed=s)
    P["agrawal_abrupt"] = lambda s: _concat(synth.Agrawal(classification_function=0, seed=s),
                                            synth.Agrawal(classification_function=4, seed=s), n)
    P["agrawal_gradual"] = lambda s: grad(synth.Agrawal(classification_function=0, seed=s),
                                          synth.Agrawal(classification_function=4, seed=s), s)
    # ---- STAGGER (3 concepts) ----
    P["stagger_nodrift"] = lambda s: synth.STAGGER(classification_function=0, seed=s)
    P["stagger_abrupt"] = lambda s: _concat(synth.STAGGER(classification_function=0, seed=s),
                                            synth.STAGGER(classification_function=2, seed=s), n)
    # ---- Sine (4 concepts) ----
    P["sine_nodrift"] = lambda s: synth.Sine(classification_function=0, seed=s)
    P["sine_abrupt"] = lambda s: _concat(synth.Sine(classification_function=0, seed=s),
                                         synth.Sine(classification_function=2, seed=s), n)
    # ---- Hyperplane (incremental drift via mag_change) ----
    P["hyperplane_static"] = lambda s: synth.Hyperplane(seed=s, n_features=10, n_drift_features=0,
                                                        mag_change=0.0)
    P["hyperplane_incr"] = lambda s: synth.Hyperplane(seed=s, n_features=10, n_drift_features=5,
                                                      mag_change=0.5)
    return P


def list_streams(n=8000):
    return list(_panel(n).keys())


def load_river_stream(name: str, n_samples: int = 8000, seed: int = 0,
                      split: str = "temporal", val_frac: float = 0.15,
                      test_frac: float = 0.15) -> TabReDDataset:
    """Materialize a named river synth stream into a TabReDDataset (binclass)."""
    try:
        import river  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("river streams need `river`: pip install river") from e
    panel = _panel(n_samples)
    if name not in panel:
        raise ValueError(f"unknown river stream {name!r}; choose from {list(panel)}")
    stream = panel[name](seed)

    feat_keys, xs, ys = None, [], []
    for i, (x, y) in enumerate(stream):
        if i >= n_samples:
            break
        if feat_keys is None:
            feat_keys = list(x.keys())
        xs.append([float(x[k]) for k in feat_keys])
        ys.append(int(bool(y)))                       # river labels are bool/0-1 here
    X_num = np.asarray(xs, dtype="float32")
    y = np.asarray(ys, dtype="int64")

    n = len(y)
    t = (np.arange(n, dtype=np.float64) / max(n - 1, 1)).astype("float32")
    t_raw = np.arange(n, dtype="int64")
    if split == "temporal":
        idx = np.arange(n)
    else:
        idx = np.random.default_rng(seed).permutation(n)
    n_te, n_va = int(test_frac * n), int(val_frac * n)
    n_tr = n - n_va - n_te
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    def mk(ii):
        return TabularSplit(X_num=X_num[ii], X_bin=None, X_cat=None,
                            y=y[ii], t=t[ii], t_raw=t_raw[ii])

    return TabReDDataset(name=f"river_{name}", task="binclass", split=split,
                         train=mk(tr), val=mk(va), test=mk(te),
                         t_min=float(t[tr].min()), t_max=float(t[tr].max()))
