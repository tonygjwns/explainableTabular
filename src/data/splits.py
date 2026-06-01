"""Split helpers, incl. our implementation of Cai & Ye (ICML 2025) temporal split.

TabReD ships these splits on disk (use tabred_loader directly):
  - 'default'            : official temporal split (train < cutoff; val next; test after)
  - 'sliding-window-{i}' : 3 sliding time windows
  - 'random-{i}'         : random shuffles within those windows (the random control)

Cai & Ye (ICML 2025) propose an IMPROVED temporal split on the same time-sorted
trainval pool (EXPERIMENT_PLAN.md §7.1):
  1. training lag = 0   : use the freshest pre-T_train data for TRAINING, not only val.
  2. validation bias min: make train->val gap mimic the train->test shift degree.
  3. direction equivalence: val may sit in the opposite temporal direction.

`cai_resplit` below is OUR concrete implementation of those principles. It operates
on the union of the 'default' train+val indices (the trainval pool), keeping test
untouched. FLAG: verify against Cai & Ye's released code before trusting numbers
(EXPERIMENT_PLAN.md §11 risk). For Phase 0 TabM reproduction, use TabReD 'default'
directly (matches published numbers); switch to cai_resplit for Phase 1+.

Pure numpy; no GPU.
"""
from __future__ import annotations

import numpy as np


def cai_resplit(
    t_raw_trainval: np.ndarray,
    val_fraction: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-split a time-sorted trainval pool per Cai & Ye principles (lag=0, bias-min).

    Args:
        t_raw_trainval: (M,) raw timestamps of the trainval pool (positions 0..M-1
            index INTO the pool, not the full dataset). Need not be pre-sorted.
        val_fraction: fraction of the pool reserved for validation.

    Returns:
        (train_pos, val_pos): integer position arrays into the trainval pool.

    Implementation (concrete, documented):
      - Sort the pool by time.
      - Validation = the MIDDLE-to-late slice sized val_fraction, positioned so its
        shift-degree to training mirrors test's shift to training, WITHOUT stealing
        the freshest block from training (lag=0 keeps the newest data in TRAIN).
      Concretely: training keeps [0 .. M*(1-val_fraction)) AND the freshest tail is
      INCLUDED in training; validation is the slice just before the tail.

    NOTE: this is a defensible operationalization, not a verbatim port. See module
    docstring. Keep it swappable.
    """
    t = np.asarray(t_raw_trainval)
    M = t.shape[0]
    order = np.argsort(t, kind="stable")        # chronological positions
    n_val = max(1, int(round(M * val_fraction)))

    # Freshest tail (size n_val) stays in TRAIN (lag=0). Validation is the block
    # immediately preceding the tail, so train->val gap ~ train->test gap.
    tail = order[M - n_val:]                     # freshest -> train
    val_pos = order[M - 2 * n_val: M - n_val]    # block before tail -> val
    train_pos = np.concatenate([order[: M - 2 * n_val], tail])
    return np.sort(train_pos), np.sort(val_pos)


def slice_indices_by_time(t_raw: np.ndarray, t_lo: float, t_hi: float) -> np.ndarray:
    """Positions where t_lo <= timestamp < t_hi. Useful for time-window analysis."""
    t = np.asarray(t_raw)
    return np.nonzero((t >= t_lo) & (t < t_hi))[0]
