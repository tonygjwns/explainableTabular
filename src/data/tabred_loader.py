"""TabReD dataset loader with time-based splits.

Encodes the evaluation protocol decided in EXPERIMENT_PLAN.md §7.1:
  - 8 TabReD datasets
  - Cai & Ye (ICML 2025) improved temporal split (training lag = 0,
    validation bias minimized) -- NOT the original TabReD split, NOT random.
  - timestamps normalized to [0, 1] over the training-available range, since
    they are used directly as the index t for the time-indexed memory P_k(t).

STATUS: skeleton. The actual loading depends on TabReD data being downloaded
(see SETUP.md §3). TODOs mark what needs the data/environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# TabReD dataset registry: name -> task type.
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

# Large datasets to subsample for feasibility (per Cai & Ye / TabReD practice).
SUBSAMPLE_TARGET: dict[str, int] = {
    "maps_routing": 200_000,
    "weather": 200_000,
    "homecredit_default": 300_000,
}


@dataclass
class TabularSplit:
    """A train/val/test split with numeric features, categorical features, labels, timestamps."""
    X_num: np.ndarray            # (N, d_num) float
    X_cat: Optional[np.ndarray]  # (N, d_cat) int or None
    y: np.ndarray                # (N,) labels/targets
    t: np.ndarray                # (N,) normalized timestamps in [0, 1]
    task: str


@dataclass
class TabReDDataset:
    name: str
    task: str
    train: TabularSplit
    val: TabularSplit
    test: TabularSplit


def normalize_time(timestamps: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """Normalize raw timestamps to [0, 1] over the training-available range.

    Used directly as the index t for P_k(t) = P_k^base + drift_k(Fourier(t)).
    Test-time t may exceed 1.0 (future) -- that's intended; Fourier extrapolates.
    """
    if t_max <= t_min:
        return np.zeros_like(timestamps, dtype=float)
    return (np.asarray(timestamps, dtype=float) - t_min) / (t_max - t_min)


def cai_temporal_split(
    timestamps: np.ndarray,
    t_train_end: float,
    val_fraction: float = 0.1,
):
    """Cai & Ye (ICML 2025) improved temporal split (EXPERIMENT_PLAN.md §7.1).

    Principle (vs original TabReD split):
      1. training lag = 0: use data right up to t_train_end for TRAINING
         (don't reserve the freshest pre-T_train data only for validation).
      2. validation bias minimized: make the train-val gap match the
         train-test gap in shift degree.

    Returns boolean masks (train_mask, val_mask). test is t > t_train_end,
    handled by the caller.

    TODO: implement exactly per Cai & Ye's released code
    (https://github.com/LAMDA-Tabular/Tabular-Temporal-Modulation).
    This stub uses a simple recent-window validation as a placeholder.
    """
    ts = np.asarray(timestamps, dtype=float)
    is_trainval = ts <= t_train_end
    # Placeholder: validation = a slice aligned to mimic test shift degree.
    # Replace with Cai & Ye's exact protocol.
    raise NotImplementedError(
        "Implement Cai & Ye (ICML 2025) split per their released code. "
        "See EXPERIMENT_PLAN.md §7.1 and PLAN.md baseline #4."
    )


def load_tabred(
    name: str,
    data_root: Path,
    subsample_seed: int = 0,
) -> TabReDDataset:
    """Load one TabReD dataset with the Cai temporal split.

    Args:
        name: key in TABRED_DATASETS.
        data_root: directory containing preprocessed TabReD data (see SETUP.md §3).
        subsample_seed: seed for subsampling large datasets.

    TODO (needs downloaded data):
      1. Read raw TabReD files (parquet/csv) from data_root / name.
      2. Apply official TabReD preprocessing (quantile transform for numeric,
         one-hot for categorical) -- reuse external/tabred preprocessing.
      3. Subsample large datasets per SUBSAMPLE_TARGET.
      4. Extract timestamps; apply cai_temporal_split + normalize_time.
      5. Return TabReDDataset.
    """
    if name not in TABRED_DATASETS:
        raise KeyError(f"Unknown TabReD dataset: {name}. Options: {list(TABRED_DATASETS)}")
    raise NotImplementedError(
        f"Download TabReD data first (SETUP.md §3), then implement loading for '{name}'. "
        "Reuse external/tabred preprocessing for fair comparison."
    )
