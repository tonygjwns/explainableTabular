"""Reproducibility: seed everything."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (if available) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Note: full determinism (torch.use_deterministic_algorithms) can slow
        # training and is not required for our seed-averaged protocol.
    except ImportError:
        pass
