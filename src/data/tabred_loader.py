"""TabReD dataset loader — reads the OFFICIAL TabReD output format directly.

Verified against yandex-research/tabred (lib/data.py, preprocessing/*.py) by
inspecting the actual repo. The TabReD preprocessing scripts write, per dataset,
under `<DATA_DIR>/<name>/`:

    X_num.npy   (float32)   numeric features        [may be absent]
    X_bin.npy   (float32)   binary features         [may be absent]
    X_cat.npy   (int64)     categorical features    [may be absent]
    X_meta.npy  (int64)     META columns; COLUMN 0 IS THE TIMESTAMP (all 8 datasets)
    Y.npy       (float32 regression / uint64 class)
    info.json   {name, task_type, score?}
    split-<split>/{train,val,test}_idx.npy   index arrays into the full arrays
    csv/        small previews

Available split names (verified): 'default' (official TEMPORAL split),
'sliding-window-0/1/2', 'random-0/1/2'. Data is time-sorted before splitting.

Timestamp: X_meta[:, 0] is the timestamp as int64 for every dataset
(sberbank/homesite/ecom='timestamp'; homecredit='date_decision'; weather='fact_time';
cooking/delivery/maps='timestamp'). We normalize it to ~[0,1] over the TRAINING
range so test-time t may exceed 1.0 (future) and Fourier terms extrapolate.

Requires the data to have been generated on the experiment machine via
`python preprocessing/<script>.py` in the tabred repo (Kaggle API; see SETUP.md §3).
This module is pure numpy and needs no GPU.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# name -> task type (matches info.json task_type values used by TabReD)
TABRED_DATASETS: dict[str, str] = {
    "sberbank_housing": "regression",
    "homesite_insurance": "binclass",
    "ecom_offers": "binclass",
    "homecredit_default": "binclass",
    "cooking_time": "regression",
    "delivery_eta": "regression",
    "maps_routing": "regression",
    "weather": "regression",
}

# Folder names as written by TabReD preprocessing scripts (hyphenated).
# Map our snake_case keys to the on-disk directory name. Adjust if your run used
# different names (check the folders created under DATA_DIR).
DIRNAME: dict[str, str] = {
    "sberbank_housing": "sberbank-housing",
    "homesite_insurance": "homesite",
    "ecom_offers": "ecom-offers",
    "homecredit_default": "homecredit",
    "cooking_time": "cooking-time",
    "delivery_eta": "delivery-eta",
    "maps_routing": "maps-routing",
    "weather": "weather",
}

# Timestamp column index within X_meta. Verified = 0 for all 8 datasets.
TIMESTAMP_META_COL: dict[str, int] = {k: 0 for k in TABRED_DATASETS}

TASK_TYPE_MAP = {"regression": "regression", "binclass": "binclass", "multiclass": "multiclass"}
_PARTS = ("train", "val", "test")


@dataclass
class TabularSplit:
    X_num: Optional[np.ndarray]   # (N, d_num) float32 or None
    X_bin: Optional[np.ndarray]   # (N, d_bin) float32 or None
    X_cat: Optional[np.ndarray]   # (N, d_cat) int64 or None
    y: np.ndarray                 # (N,)
    t: np.ndarray                 # (N,) normalized timestamps (train range -> [0,1])
    t_raw: np.ndarray             # (N,) raw int64 timestamps (for inspection)


@dataclass
class TabReDDataset:
    name: str
    task: str                     # binclass | multiclass | regression
    split: str                    # which split was loaded
    train: TabularSplit
    val: TabularSplit
    test: TabularSplit
    t_min: float                  # training-range min raw timestamp (for normalization)
    t_max: float                  # training-range max raw timestamp


def _load_npy(path: Path) -> Optional[np.ndarray]:
    return np.load(path, allow_pickle=False) if path.exists() else None


def normalize_time(raw: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """Normalize raw int64 timestamps to [0,1] over [t_min, t_max] (training range).

    Test timestamps > t_max map to > 1.0 by design (future extrapolation).
    """
    raw = np.asarray(raw, dtype=np.float64)
    if t_max <= t_min:
        return np.zeros_like(raw, dtype=np.float32)
    return ((raw - t_min) / (t_max - t_min)).astype(np.float32)


def available_splits(name: str, data_root: Path) -> list[str]:
    """List split names present on disk for a dataset (e.g. default, random-0...)."""
    base = Path(data_root) / DIRNAME[name]
    return sorted(p.name[len("split-"):] for p in base.glob("split-*") if p.is_dir())


def load_tabred(
    name: str,
    data_root: Path,
    split: str = "default",
) -> TabReDDataset:
    """Load one TabReD dataset for a given split, with normalized timestamps.

    Args:
        name: key in TABRED_DATASETS (snake_case).
        data_root: directory containing the per-dataset folders (TabReD DATA_DIR).
        split: 'default' (temporal), 'random-{0,1,2}', or 'sliding-window-{0,1,2}'.
               Phase 0 reproduction uses 'default' to match published numbers.
    Returns:
        TabReDDataset with train/val/test TabularSplit and normalized t.
    """
    if name not in TABRED_DATASETS:
        raise KeyError(f"Unknown TabReD dataset: {name}. Options: {list(TABRED_DATASETS)}")
    base = Path(data_root) / DIRNAME[name]
    if not base.exists():
        raise FileNotFoundError(
            f"{base} not found. Generate it on the experiment machine via "
            f"`python preprocessing/{DIRNAME[name]}.py` in the tabred repo (SETUP.md §3)."
        )

    info = json.loads((base / "info.json").read_text())
    task = TASK_TYPE_MAP[info["task_type"]]

    # full arrays (may be absent for a given feature kind)
    X_num_full = _load_npy(base / "X_num.npy")
    X_bin_full = _load_npy(base / "X_bin.npy")
    X_cat_full = _load_npy(base / "X_cat.npy")
    X_meta_full = _load_npy(base / "X_meta.npy")
    Y_full = _load_npy(base / "Y.npy")
    if X_meta_full is None:
        raise FileNotFoundError(f"{base}/X_meta.npy missing — timestamp comes from it.")
    if Y_full is None:
        raise FileNotFoundError(f"{base}/Y.npy missing.")

    ts_col = TIMESTAMP_META_COL[name]
    t_full = np.asarray(X_meta_full[:, ts_col]).astype(np.int64)

    split_dir = base / f"split-{split}"
    if not split_dir.exists():
        raise FileNotFoundError(
            f"{split_dir} not found. Available: {available_splits(name, data_root)}"
        )
    idx = {p: np.load(split_dir / f"{p}_idx.npy", allow_pickle=False) for p in _PARTS}

    # Normalization range from TRAINING timestamps only (our convention).
    t_train_raw = t_full[idx["train"]]
    t_min, t_max = float(t_train_raw.min()), float(t_train_raw.max())

    def make(part: str) -> TabularSplit:
        ii = idx[part]
        return TabularSplit(
            X_num=None if X_num_full is None else X_num_full[ii],
            X_bin=None if X_bin_full is None else X_bin_full[ii],
            X_cat=None if X_cat_full is None else X_cat_full[ii],
            y=Y_full[ii],
            t=normalize_time(t_full[ii], t_min, t_max),
            t_raw=t_full[ii],
        )

    return TabReDDataset(
        name=name, task=task, split=split,
        train=make("train"), val=make("val"), test=make("test"),
        t_min=t_min, t_max=t_max,
    )


if __name__ == "__main__":
    # Usage probe (no data on this machine; just prints the contract).
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--dataset", default="sberbank_housing")
    ap.add_argument("--split", default="default")
    a = ap.parse_args()
    ds = load_tabred(a.dataset, Path(a.data_root), a.split)
    print(f"{ds.name} [{ds.task}] split={ds.split}")
    for part in ("train", "val", "test"):
        s = getattr(ds, part)
        nn = None if s.X_num is None else s.X_num.shape
        print(f"  {part}: y={s.y.shape} X_num={nn} t in [{s.t.min():.3f},{s.t.max():.3f}]")
