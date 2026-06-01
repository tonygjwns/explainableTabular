"""TabM backbone wrapper — written against the VERIFIED official API.

Inspected yandex-research/tabm (tabm.py) directly. Key facts:
  - Factory: TabM.make(n_num_features=, cat_cardinalities=[...], d_out=, k=32,
             n_blocks=, d_block=, dropout=, activation='ReLU', arch_type='tabm',
             num_embeddings=None, ...)
  - forward(x_num, x_cat) -> (batch, k, d_out if d_out else d_block)
      * d_out=None  -> representation, k-submodel axis PRESERVED  (B, k, d_block)
      * d_out=C     -> per-submodel predictions                   (B, k, C)
  - Input: x_num (B, n_num) float, x_cat (B, n_cat) long; cat one-hot encoded internally.
  - num_embeddings: optional PLR/PLE (rtdl_num_embeddings) — used in Phase 2, not Phase 0.

Two uses:
  - Phase 0 baseline: d_out=C, `predict()` averages the k submodel predictions
    (mean of softmax probs for classification; mean for regression).
  - Phase 1: d_out=None, `encode()` returns the representation. Per HANDOFF Don'ts,
    do NOT average the k axis and then call it TabM-T as the *final* design; for the
    Phase-1 minimal version a single representation (reduce='mean') is the documented
    starting point, and per-submodel integration is a Phase-2 ablation (EXPERIMENT_PLAN §9).

Install TabM from PyPI (it is a single-file library, do NOT git-clone+`-e`):
    pip install tabm                    # then `from tabm import TabM`
The sys.path fallback below is only for the rare case of using a local checkout.

Requires PyTorch + numpy + the tabm package. (Common gotcha: installing torch but
forgetting numpy -> `from tabm import TabM` works but everything else dies with
"Failed to initialize NumPy". Use requirements.txt.)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Try to import the official TabM; fall back to a conventional external/ path.
try:
    from tabm import TabM  # type: ignore
except ImportError:  # pragma: no cover - depends on machine setup
    _candidates = [
        Path(os.environ.get("TABM_REPO", "")),
        Path.home() / "Desktop" / "external" / "tabm",
        Path(__file__).resolve().parents[2].parent / "external" / "tabm",
    ]
    for _c in _candidates:
        if _c and (_c / "tabm.py").exists():
            sys.path.insert(0, str(_c))
            break
    from tabm import TabM  # type: ignore  # noqa: E402


def compute_cat_cardinalities(x_cat_train: Optional[np.ndarray]) -> list[int]:
    """cat_cardinalities[i] = number of categories of feature i (from TRAIN split).

    TabReD X_cat is int64 with categories in range(0, cardinality). We use
    train max + 1 per column. (TabReD's own pipeline maps unseen test categories
    to an unknown bucket; keep that behavior if you reuse their CatPolicy.)
    """
    if x_cat_train is None or x_cat_train.size == 0:
        return []
    return [int(x_cat_train[:, j].max()) + 1 for j in range(x_cat_train.shape[1])]


class TabMBackbone(torch.nn.Module):
    """Thin wrapper over official TabM.

    Args:
        n_num_features: numeric feature count (INCLUDE binary features here if you
            concatenate X_bin into x_num, which is the recommended handling).
        cat_cardinalities: from compute_cat_cardinalities(train X_cat).
        d_out: None -> representation mode (encode); int -> prediction mode (predict).
        k, n_blocks, d_block, dropout, arch_type: TabM hyperparameters.
        num_embeddings: optional PLR/PLE module (Phase 2).
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int],
        d_out: Optional[int],
        k: int = 32,
        n_blocks: int = 3,
        d_block: int = 512,
        dropout: float = 0.1,
        arch_type: str = "tabm",
        num_embeddings=None,
    ):
        super().__init__()
        self.d_out = d_out
        self.k = k
        self.model = TabM.make(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_out=d_out,
            k=k,
            n_blocks=n_blocks,
            d_block=d_block,
            dropout=dropout,
            arch_type=arch_type,
            num_embeddings=num_embeddings,
        )

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        """Raw TabM output: (B, k, d_out or d_block)."""
        return self.model(x_num, x_cat)

    def encode(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
        reduce: str = "mean",
    ) -> torch.Tensor:
        """Representation for the memory+retrieval layer (requires d_out=None).

        reduce='mean' -> (B, d_block)  [Phase-1 minimal: single representation]
        reduce='none' -> (B, k, d_block)  [Phase-2: per-submodel integration]
        """
        if self.d_out is not None:
            raise RuntimeError("encode() requires the backbone built with d_out=None.")
        z = self.model(x_num, x_cat)            # (B, k, d_block)
        if reduce == "mean":
            return z.mean(dim=1)                # (B, d_block)
        if reduce == "none":
            return z                            # (B, k, d_block)
        raise ValueError(f"reduce must be 'mean' or 'none', got {reduce}")

    @torch.no_grad()
    def predict(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
        task: str,
    ) -> torch.Tensor:
        """Phase-0 baseline prediction: average the k submodel outputs.

        Classification -> mean of per-submodel softmax probs -> (B, C).
        Regression     -> mean over k -> (B,).
        (requires d_out set to C for classification or 1 for regression)
        """
        if self.d_out is None:
            raise RuntimeError("predict() requires the backbone built with d_out set.")
        out = self.model(x_num, x_cat)          # (B, k, C) or (B, k, 1)
        task = task.lower()
        if task in {"binclass", "multiclass"}:
            probs = F.softmax(out, dim=-1)       # (B, k, C)
            return probs.mean(dim=1)             # (B, C)
        # regression
        return out.mean(dim=1).squeeze(-1)       # (B,)
