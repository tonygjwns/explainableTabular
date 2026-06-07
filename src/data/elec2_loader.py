"""Elec2 (Electricity) loader — the canonical real concept-drift tabular benchmark.

~45k half-hourly records; binary target (price UP/DOWN vs a moving average); the
feature->label relationship shifts over time (market regimes) = real concept drift.
Used as the DECIDER: does the (synthetic-validated) time-indexed memory beat a fixed
memory on REAL concept drift? (Bridges synthetic vs TabReD.)

Returns a TabReDDataset so the existing Phase-1 trainer works unchanged.
- t = stream position normalized to [0,1] (the temporal coordinate the memory indexes).
- split='random' (all t covered in train; cleanest "exploits concept drift" test,
  matching the synthetic control) or 'temporal' (train=past, test=future; harder
  extrapolation, like TabReD).

Source: sklearn.datasets.fetch_openml(data_id=151) (cached under ~/scikit_learn_data).
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml

from .tabred_loader import TabularSplit, TabReDDataset

NUM_COLS = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer", "period"]
CAT_COLS = ["day"]


def load_elec2(split: str = "random", seed: int = 0,
               val_frac: float = 0.15, test_frac: float = 0.15) -> TabReDDataset:
    ds = fetch_openml(data_id=151, as_frame=True)
    df = ds.frame.copy()

    # target -> {0,1}; classes are 'UP'/'DOWN' (or already 0/1)
    yraw = df[ds.target_names[0]] if hasattr(ds, "target_names") and ds.target_names else df["class"]
    ys = yraw.astype(str).str.upper()
    y = (ys == "UP").astype("int64").to_numpy()
    if set(np.unique(ys)) - {"UP", "DOWN"}:           # fallback if not UP/DOWN coded
        y = yraw.astype("category").cat.codes.to_numpy().astype("int64")

    num_cols = [c for c in NUM_COLS if c in df.columns]
    X_num = df[num_cols].astype("float32").to_numpy()
    cat_cols = [c for c in CAT_COLS if c in df.columns]
    if cat_cols:
        X_cat = np.empty((len(df), len(cat_cols)), dtype="int64")
        for j, c in enumerate(cat_cols):
            codes = df[c].astype("category").cat.codes.to_numpy()   # 0..k-1
            X_cat[:, j] = codes
    else:
        X_cat = None

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
        return TabularSplit(X_num=X_num[ii], X_bin=None,
                            X_cat=None if X_cat is None else X_cat[ii],
                            y=y[ii], t=t[ii], t_raw=t_raw[ii])

    return TabReDDataset(name="elec2", task="binclass", split=split,
                         train=mk(tr), val=mk(va), test=mk(te),
                         t_min=float(t[tr].min()), t_max=float(t[tr].max()))
