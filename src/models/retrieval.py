"""Retrieval over the time-indexed prototype memory + the full memory layer.

Phase-1 minimal retrieval (decision 1, EXPERIMENT_PLAN.md §3, §6):
    w_k = softmax( -||z - P_k(t_x)||^2 / tau_temp )   over k
    aggregated = sum_k w_k * V_k
NO WTA, NO TabR correction term, NO outer-product gating (those are Phase-2
ablations, decision 3).

`MemoryRetrievalLayer` composes:
    backbone representation z  +  time t  ->  aggregated memory readout
and a small predictor maps [z ; aggregated] -> y_hat.

This layer sits BETWEEN the (frozen-in-Phase-1 or jointly-trained) backbone and
the output -- it is the memory+retrieval layer, not a head bolted on logits.

Requires PyTorch.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prototype_memory import TimeIndexedPrototypeMemory
from .value_module import ValueModule


def softmax_retrieval(
    z: torch.Tensor,            # (B, d)
    prototypes: torch.Tensor,   # (B, K, d)   P_k(t_x) per sample
    tau_temp: float = 1.0,
):
    """Return retrieval weights w (B, K) and squared distances (B, K)."""
    # ||z - P_k||^2 = ||z||^2 - 2 z.P_k + ||P_k||^2 ; compute directly for clarity
    diff = z.unsqueeze(1) - prototypes            # (B, K, d)
    sq_dist = diff.pow(2).sum(dim=-1)             # (B, K)
    w = F.softmax(-sq_dist / tau_temp, dim=-1)    # (B, K)
    return w, sq_dist


class MemoryRetrievalLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        memory: TimeIndexedPrototypeMemory,
        value_module: ValueModule,
        out_dim: int,
        tau_temp: float = 1.0,
        predictor_hidden: int = 256,
        predictor_mode: str = "concat",
    ):
        """
        Args:
            dim: representation dim d.
            memory: TimeIndexedPrototypeMemory (holds P_k(t)).
            value_module: ValueModule (holds V_k).
            out_dim: prediction output dim (1 for binclass/regression, C for multiclass).
            tau_temp: softmax temperature for retrieval.
            predictor_hidden: hidden width of the predictor MLP(s).
            predictor_mode: how the memory readout reaches the output —
                "concat"      : MLP([z ; aggregated])           (memory can be bypassed via z)
                "memory_only" : MLP(aggregated)                 (forces all signal through memory)
                "residual"    : head_z(z) + head_mem(aggregated) (explicit, separable branches)
        """
        super().__init__()
        self.memory = memory
        self.value_module = value_module
        self.tau_temp = tau_temp
        self.predictor_mode = predictor_mode

        def mlp(in_dim):
            return nn.Sequential(nn.Linear(in_dim, predictor_hidden), nn.ReLU(),
                                 nn.Linear(predictor_hidden, out_dim))

        if predictor_mode == "concat":
            self.predictor = mlp(2 * dim)
        elif predictor_mode == "memory_only":
            self.predictor = mlp(dim)
        elif predictor_mode == "residual":
            self.head_z = mlp(dim)
            self.head_mem = mlp(dim)
        else:
            raise ValueError(f"Unknown predictor_mode: {predictor_mode}")

    def forward(self, z: torch.Tensor, t: torch.Tensor, return_aux: bool = False,
                ablate_memory: bool = False):
        """
        Args:
            z: (B, d) backbone representation.
            t: (B,) normalized timestamps.
            return_aux: if True, also return retrieval weights & distances (for
                        Sanity Test 3 retrieval-concentration analysis).
            ablate_memory: if True, zero the aggregated memory readout (predictor
                        sees [z ; 0]). Used by diagnostics to measure how much the
                        memory actually contributes (mem_gap).
        Returns:
            y_hat: (B, out_dim). If return_aux: (y_hat, {"w":..., "sq_dist":...}).
        """
        prototypes = self.memory.prototypes_at(t)            # (B, K, d)
        w, sq_dist = softmax_retrieval(z, prototypes, self.tau_temp)  # (B, K)
        V = self.value_module.values()                        # (K, d)
        aggregated = w @ V                                    # (B, d)
        if ablate_memory:
            aggregated = torch.zeros_like(aggregated)
        if self.predictor_mode == "concat":
            y_hat = self.predictor(torch.cat([z, aggregated], dim=-1))
        elif self.predictor_mode == "memory_only":
            y_hat = self.predictor(aggregated)
        else:  # residual
            y_hat = self.head_z(z) + self.head_mem(aggregated)
        if return_aux:
            return y_hat, {"w": w, "sq_dist": sq_dist}
        return y_hat

    def smoothness_penalty(self, t: torch.Tensor) -> torch.Tensor:
        """Convenience passthrough to memory.smoothness_penalty (L_smooth)."""
        return self.memory.smoothness_penalty(t)
