"""Fourier time embedding tau(t).

Implements the time encoding used as the index for the time-indexed prototype
memory (EXPERIMENT_PLAN.md §3, §5). Follows Cai & Ye (ICML 2025) Fourier-series
form with pre-defined periodicity priors + an optional linear trend term:

    tau(t) = [ ReLU(Linear(Periodic(t))) , Trend(t) ]
    Periodic(t) = concat_i Fourier(t, T_i)
    Fourier(t, T) = [ sin(2*pi*k*t/T), cos(2*pi*k*t/T) ]  for k = 1..K
    Trend(t) = standardized t  (optional)

NOTE: pre-defined periods (yearly/monthly/weekly/daily) are more stable and
interpretable than learned frequencies (we cite Cai & Ye for this choice).
`t` is expected normalized to ~[0, 1] over the training-available range; test-time
t may exceed 1.0 (future) and Fourier terms extrapolate by construction.

Requires PyTorch (run on the server; not installed on the doc machine).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class FourierTimeEmbedding(nn.Module):
    def __init__(
        self,
        out_dim: int,
        periods: Sequence[float] = (1.0, 1.0 / 12, 1.0 / 52, 1.0 / 365),
        n_harmonics: int = 6,
        use_trend: bool = True,
    ):
        """
        Args:
            out_dim: dimension of the periodic projection output (the ReLU(Linear(.)) part).
                     Final embedding dim = out_dim + (1 if use_trend else 0).
            periods: periodicity priors T_i in the SAME units as the (normalized) t.
                     Defaults assume t normalized so that 1.0 == one year, hence
                     yearly=1, monthly=1/12, weekly=1/52, daily=1/365.
                     ADJUST to your timestamp normalization (see tabred_loader.normalize_time).
            n_harmonics: number of Fourier harmonics K per period.
            use_trend: append standardized t to capture linear trend.
        """
        super().__init__()
        self.periods = list(periods)
        self.n_harmonics = int(n_harmonics)
        self.use_trend = bool(use_trend)

        # raw Fourier feature count: 2 (sin,cos) * K harmonics * n_periods
        raw_dim = 2 * self.n_harmonics * len(self.periods)
        self.proj = nn.Linear(raw_dim, out_dim)
        self.act = nn.ReLU()
        self.out_dim = out_dim + (1 if self.use_trend else 0)

        # harmonic multipliers k = 1..K, registered as buffer (not trained)
        k = torch.arange(1, self.n_harmonics + 1, dtype=torch.float32)
        self.register_buffer("harmonics", k, persistent=False)

    def _raw_fourier(self, t: torch.Tensor) -> torch.Tensor:
        """t: (batch,) -> (batch, raw_dim)."""
        t = t.reshape(-1, 1)                                   # (B, 1)
        feats = []
        for T in self.periods:
            # angle: (B, K) = 2*pi*k*t / T
            ang = 2.0 * torch.pi * self.harmonics.reshape(1, -1) * t / float(T)
            feats.append(torch.sin(ang))
            feats.append(torch.cos(ang))
        return torch.cat(feats, dim=1)                          # (B, raw_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (batch,) normalized timestamps -> tau: (batch, out_dim)."""
        raw = self._raw_fourier(t)                              # (B, raw_dim)
        periodic = self.act(self.proj(raw))                     # (B, out_dim_periodic)
        if self.use_trend:
            trend = t.reshape(-1, 1)                            # already ~standardized
            return torch.cat([periodic, trend], dim=1)          # (B, out_dim)
        return periodic
