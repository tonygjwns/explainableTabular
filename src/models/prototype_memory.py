"""Time-indexed prototype memory: P_k(t) = P_k^base + drift_k(tau(t)).

The core novel component (EXPERIMENT_PLAN.md §3). Stores K prototypes whose
positions evolve smoothly over time. This is a MEMORY (stores time-distribution
info), NOT a prediction head — input is retrieved against it (see retrieval.py).

Design (Phase-1 minimal, EXPERIMENT_PLAN §6):
  - P_k^base: (K, d) learnable. Initialized by time-sliced KMeans (decision 5;
    see `init_from_kmeans`), NOT random.
  - drift_k(tau(t)): per-prototype offset as a function of a SHARED time code.
    Parameter-efficient form: a shared MLP maps tau(t) -> h(t) in R^r, then a
    per-prototype low-rank projection W_k in R^{d x r} gives the offset:
        Delta_k(t) = W_k @ h(t)
    So each prototype gets its OWN drift direction driven by a shared time code.
    Params: K*d*r  (e.g. K=1000, d=256, r=32 -> ~8M).
  - `time_indexed=False` -> drift disabled -> P_k(t) = P_k^base (the "Fixed
    memory" sanity-check control, Test 1).

WTA / annealing are intentionally absent in Phase 1 (decision 3).

Requires PyTorch.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .temporal_embedding import FourierTimeEmbedding


class TimeIndexedPrototypeMemory(nn.Module):
    def __init__(
        self,
        n_prototypes: int,
        dim: int,
        time_embedding: FourierTimeEmbedding,
        rank: int = 32,
        hidden: int = 64,
        time_indexed: bool = True,
    ):
        """
        Args:
            n_prototypes: K.
            dim: prototype/representation dimension d.
            time_embedding: a FourierTimeEmbedding instance (shared, can be reused).
            rank: r, low-rank size of the per-prototype drift projection.
            hidden: hidden width of the shared time-code MLP.
            time_indexed: if False, behaves as fixed memory (control for Test 1).
        """
        super().__init__()
        self.K = n_prototypes
        self.d = dim
        self.time_indexed = time_indexed
        self.time_emb = time_embedding

        self.P_base = nn.Parameter(torch.zeros(n_prototypes, dim))
        nn.init.normal_(self.P_base, std=0.02)

        if time_indexed:
            # shared time-code MLP: tau(t) -> h(t) in R^r
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embedding.out_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, rank),
            )
            # per-prototype low-rank projection W in (K, d, r)
            self.W = nn.Parameter(torch.zeros(n_prototypes, dim, rank))
            nn.init.normal_(self.W, std=0.02 / max(1, rank) ** 0.5)
            self.rank = rank

    @torch.no_grad()
    def init_from_kmeans(self, base: np.ndarray) -> None:
        """Initialize P_base from time-sliced KMeans centroids (decision 5).

        Args:
            base: (K, d) numpy array of centroids (caller runs KMeans per time
                  slice and stacks/aligns to K rows). See EXPERIMENT_PLAN §6.
        """
        base = np.asarray(base, dtype=np.float32)
        if base.shape != (self.K, self.d):
            raise ValueError(f"expected ({self.K},{self.d}), got {base.shape}")
        self.P_base.copy_(torch.from_numpy(base))

    def prototypes_at(self, t: torch.Tensor) -> torch.Tensor:
        """Return P_k(t) for each sample's timestamp.

        Args:
            t: (batch,) normalized timestamps.
        Returns:
            (batch, K, d) prototype positions at each sample's time.
        """
        B = t.shape[0]
        if not self.time_indexed:
            return self.P_base.unsqueeze(0).expand(B, self.K, self.d)

        tau = self.time_emb(t)                       # (B, time_dim)
        h = self.time_mlp(tau)                        # (B, r)
        # Delta[b,k,:] = W[k] @ h[b]  -> einsum over r
        delta = torch.einsum("kdr,br->bkd", self.W, h)  # (B, K, d)
        return self.P_base.unsqueeze(0) + delta          # (B, K, d)

    def smoothness_penalty(self, t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """L_smooth: penalize drift change between nearby times (EXPERIMENT_PLAN §5).

        Finite-difference proxy for ||dP/dt||^2, evaluated at each sample's t and
        a slightly perturbed t+eps, averaged over batch and prototypes.
        Only meaningful when time_indexed=True (else returns 0).
        """
        if not self.time_indexed:
            return t.new_zeros(())
        p1 = self.prototypes_at(t)              # (B, K, d)
        p2 = self.prototypes_at(t + eps)        # (B, K, d)
        diff = (p2 - p1) / eps                   # approx dP/dt
        return diff.pow(2).mean()
