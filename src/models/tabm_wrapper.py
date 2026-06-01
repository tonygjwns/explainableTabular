"""TabM backbone wrapper.

TabM (Gorishniy et al., ICLR 2025) is our backbone (EXPERIMENT_PLAN.md §3).
This wraps the official TabM implementation so the rest of our pipeline can
treat it as: x -> z (representation) and/or x -> y_hat (prediction).

KEY CONSTRAINT (do NOT violate): TabM produces k=32 implicit submodels via
BatchEnsemble. Do NOT average the 32 submodels before attaching our memory +
retrieval layer -- that destroys TabM's ensemble diversity (see HANDOFF Don'ts).
Phase 1 uses a single backbone representation; the 32-submodel integration is a
Phase 2 ablation (EXPERIMENT_PLAN.md §9).

STATUS: skeleton. Requires external/tabm to be cloned (SETUP.md §2).
"""
from __future__ import annotations

from typing import Optional

# import torch
# import torch.nn as nn
# Add external/tabm to PYTHONPATH or pip install -e it, then:
# from tabm import TabM   # exact import path per the official repo


class TabMBackbone:
    """Thin wrapper around official TabM.

    Intended interface:
        backbone = TabMBackbone(n_num_features, cat_cardinalities, d_out, k=32, ...)
        z = backbone.encode(x_num, x_cat)        # (batch, k, d) or (batch, d)
        y = backbone.predict(x_num, x_cat)       # standard TabM head (Phase 0 baseline)

    TODO (needs external/tabm):
      1. Import the official TabM class.
      2. Map our config (configs/tabm_baseline.yaml) to TabM constructor args.
      3. Implement encode() that returns the representation BEFORE the prediction
         head, preserving the k-submodel axis (do not average prematurely).
      4. Implement predict() for the Phase 0 reproduction baseline.
    """

    def __init__(self, config: dict):
        self.config = config
        raise NotImplementedError(
            "Clone external/tabm (SETUP.md §2) and wire up the official TabM class here. "
            "Preserve the k=32 submodel axis in encode() -- do not average early."
        )

    def encode(self, x_num, x_cat=None):
        raise NotImplementedError

    def predict(self, x_num, x_cat=None):
        raise NotImplementedError
