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
        basis: str = "fourier",
        trend_degree: int = 3,
    ):
        """
        Args:
            out_dim: dimension of the projected output (the ReLU(Linear(.)) part).
                     Final embedding dim = out_dim + (1 if use_trend else 0).
            periods/n_harmonics: Fourier params (basis='fourier').
            use_trend: append raw t (linear trend term).
            basis: 'fourier' (periodic; oscillates for t>1 -> bad extrapolation) or
                   'trend' (polynomial [t, t^2, ..., t^deg]; monotone, extrapolation-safe).
                   Use 'trend' for the rescue plan's extrapolation regime.
            trend_degree: polynomial degree for basis='trend'.
        """
        super().__init__()
        self.periods = list(periods)
        self.n_harmonics = int(n_harmonics)
        self.use_trend = bool(use_trend)
        self.basis = basis
        self.trend_degree = int(trend_degree)

        if basis == "fourier":
            raw_dim = 2 * self.n_harmonics * len(self.periods)
            k = torch.arange(1, self.n_harmonics + 1, dtype=torch.float32)
            self.register_buffer("harmonics", k, persistent=False)
        elif basis == "trend":
            raw_dim = self.trend_degree                        # [t, t^2, ..., t^deg]
            powers = torch.arange(1, self.trend_degree + 1, dtype=torch.float32)
            self.register_buffer("powers", powers, persistent=False)
        else:
            raise ValueError(f"Unknown basis: {basis}")

        self.proj = nn.Linear(raw_dim, out_dim)
        self.act = nn.ReLU()
        self.out_dim = out_dim + (1 if self.use_trend else 0)

    def _raw(self, t: torch.Tensor) -> torch.Tensor:
        """t: (batch,) -> (batch, raw_dim)."""
        t = t.reshape(-1, 1)                                   # (B, 1)
        if self.basis == "fourier":
            feats = []
            for T in self.periods:
                ang = 2.0 * torch.pi * self.harmonics.reshape(1, -1) * t / float(T)
                feats.append(torch.sin(ang)); feats.append(torch.cos(ang))
            return torch.cat(feats, dim=1)
        # trend: polynomial powers (monotone, extrapolation-safe)
        return t ** self.powers.reshape(1, -1)                 # (B, deg)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (batch,) normalized timestamps -> tau: (batch, out_dim)."""
        proj = self.act(self.proj(self._raw(t)))               # (B, out_dim_proj)
        if self.use_trend:
            return torch.cat([proj, t.reshape(-1, 1)], dim=1)  # (B, out_dim)
        return proj
