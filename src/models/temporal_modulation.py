"""Faithful re-implementation of Cai & Ye (NeurIPS 2025) temporal feature modulation.

Used by R2.3 (the adjudication, PLAN_V2 §R2): their "Feature-aware Modulation for
Learning from Temporal Tabular Data" modulates INPUT feature distributional
statistics as a function of time and reports it beats baselines on TabReD. Reading
their model/lib/temporal_modulation.py, the mechanism is (verbatim shape):

    gamma  = fc_gamma(time_emb(t))     # per-feature scale
    beta   = fc_beta(time_emb(t))      # per-feature shift
    lambda = fc_lambda(time_emb(t))    # per-feature Yeo-Johnson power
    x = yeo_johnson(x, lambda)         # reshape skew/distribution of x
    x = gamma * x + beta               # affine (mean/scale) align

CRITICAL for the adjudication: every parameter is a function of t and x ONLY —
NOTHING depends on the label y. So this is a time-INDEXED COVARIATE (P(x))
normalization by construction; it cannot exploit a P(y|x) (concept) change. Hence
ANY gain it yields on a dataset with measured concept≈0 (cooking/maps, RESULTS §13)
is X-side adaptation — the empirical half of the adjudication. (The definitional
half is this docstring + the code.)

Differentiable Yeo-Johnson (handles x<0; stable near lambda∈{0,2}). Requires PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .temporal_embedding import FourierTimeEmbedding


def yeo_johnson(x: torch.Tensor, lam: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Element-wise Yeo-Johnson power transform, differentiable in x and lam.

    x>=0:  ((x+1)^lam - 1)/lam            (lam->0: log(x+1))
    x<0:  -(((-x+1)^(2-lam) - 1)/(2-lam)) (lam->2: -log(-x+1))
    Branches blended by the sign mask; lam denominators are eps-guarded so the
    transform stays finite and smooth across lam=0 / lam=2.
    """
    pos = (x >= 0).to(x.dtype)
    xp = torch.clamp(x, min=0.0)
    xn = torch.clamp(-x, min=0.0)
    lam_safe = torch.where(lam.abs() < eps, torch.full_like(lam, eps), lam)
    two_lam = 2.0 - lam
    two_lam_safe = torch.where(two_lam.abs() < eps, torch.full_like(two_lam, eps), two_lam)
    yp = (torch.pow(xp + 1.0, lam) - 1.0) / lam_safe
    yn = -(torch.pow(xn + 1.0, two_lam) - 1.0) / two_lam_safe
    return pos * yp + (1.0 - pos) * yn


class TemporalModulation(nn.Module):
    """Time-conditioned per-feature Yeo-Johnson + affine modulation of input x.

    A faithful, label-free reimplementation of the Cai & Ye (NeurIPS 2025) module.
    gamma/beta/lambda are LINEAR maps of a time embedding (zero-init on the heads so
    the layer starts as the identity transform: gamma=1, beta=0, lambda=1 -> x↦x).
    """

    def __init__(self, n_features: int, *, time_out: int = 16,
                 time_basis: str = "trend", trend_degree: int = 3):
        super().__init__()
        self.time_emb = FourierTimeEmbedding(time_out, basis=time_basis,
                                             trend_degree=trend_degree, use_trend=True)
        td = self.time_emb.out_dim
        self.fc_gamma = nn.Linear(td, n_features)
        self.fc_beta = nn.Linear(td, n_features)
        self.fc_lambda = nn.Linear(td, n_features)
        # identity at init: gamma=1, beta=0, lambda=1 (Yeo-Johnson with lam=1 is x↦x)
        nn.init.zeros_(self.fc_gamma.weight); nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight); nn.init.zeros_(self.fc_beta.bias)
        nn.init.zeros_(self.fc_lambda.weight); nn.init.ones_(self.fc_lambda.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x (B, n_features), t (B,) -> modulated x (B, n_features)."""
        te = self.time_emb(t)
        gamma = self.fc_gamma(te)
        beta = self.fc_beta(te)
        lam = self.fc_lambda(te).clamp(-2.0, 4.0)     # keep YJ powers in a sane range
        return gamma * yeo_johnson(x, lam) + beta
