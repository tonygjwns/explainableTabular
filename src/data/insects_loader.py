"""INSECTS loader — a SECOND real concept-drift benchmark (designed drift, multiclass).

elec2 gave a robust NEGATIVE for the time-TabR *structure* (RESULTS §10), but elec2 is
a trivially-exploitable autocorrelated stream (a plain time feature captures its concept).
The pre-registered decision rule needs >=2 datasets. INSECTS is the complement:
- optical-sensor wing-beat features (33 numeric), target = insect SPECIES (multiclass),
- drift is DESIGNED (the stream is generated under controlled temperature changes), with
  named regimes: abrupt / incremental / incremental-reoccurring / gradual,
- NOT subject to the elec2 autocorrelation-leakage critique → a cleaner test of whether
  the time-TabR STRUCTURE beats a time-FEATURE when concept is non-trivial.

Returns a TabReDDataset so the Q2b trainer/runner work unchanged (task='multiclass';
the trainer + TimeTabRModel already support multiclass — covered by the smoke tests).
- t = stream position normalized to [0,1] (the temporal coordinate; the stream order
  IS the drift trajectory).
- split='temporal' (train=past, test=future; the meaningful drift test) or 'random'.

Source: `river.datasets.Insects(variant=...)` (downloads + caches on first use).
Requires the `river` package (`pip install river`); install on the server.
"""
from __future__ import annotations

import numpy as np

from .tabred_loader import TabularSplit, TabReDDataset

# river variant names (see river.datasets.Insects). 'incremental_balanced' is the
# canonical gradual-drift, class-balanced stream — a clean default for Q2b.
# NOTE: river 0.25.0 accepts exactly these seven (server ValueError 2026-07-04 listed them);
# older river versions exposed more *_imbalanced combos and out_of_control.
VARIANTS = (
    "abrupt_balanced", "abrupt_imbalanced",
    "gradual_balanced", "gradual_imbalanced",
    "incremental_balanced",
    "incremental_abrupt_balanced", "incremental_reoccurring_balanced",
)


def load_insects(variant: str = "incremental_balanced", split: str = "temporal",
                 seed: int = 0, val_frac: float = 0.15, test_frac: float = 0.15,
                 max_samples: int | None = None) -> TabReDDataset:
    """Load an INSECTS stream into a TabReDDataset (multiclass).

    max_samples: cap the stream length (head) for faster experiments; None = full.
    """
    try:
        from river import datasets
    except ImportError as e:  # pragma: no cover
        raise ImportError("INSECTS needs `river`: pip install river") from e
    if variant not in VARIANTS:
        raise ValueError(f"unknown INSECTS variant {variant!r}; choose from {VARIANTS}")

    stream = datasets.Insects(variant=variant)
    feat_keys = None
    xs, ys = [], []
    for i, (x, y) in enumerate(stream):
        if max_samples is not None and i >= max_samples:
            break
        if feat_keys is None:
            feat_keys = list(x.keys())          # fix column order from the first sample
        xs.append([x[k] for k in feat_keys])
        ys.append(y)

    X_num = np.asarray(xs, dtype="float32")
    classes = sorted(set(ys))                   # deterministic class->index map
    cls_index = {c: j for j, c in enumerate(classes)}
    y = np.asarray([cls_index[c] for c in ys], dtype="int64")

    n = len(y)
    t = (np.arange(n, dtype=np.float64) / max(n - 1, 1)).astype("float32")  # stream order
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

    return TabReDDataset(name=f"insects_{variant}", task="multiclass", split=split,
                         train=mk(tr), val=mk(va), test=mk(te),
                         t_min=float(t[tr].min()), t_max=float(t[tr].max()))
