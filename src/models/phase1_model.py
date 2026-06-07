"""Phase-1 end-to-end model: TabM backbone -> time-indexed memory + retrieval.

Assembles the already-implemented novel modules ON TOP of the TabM backbone's
representation, as the MINIMAL Phase-1 design (EXPERIMENT_PLAN.md §6):

  1. Backbone: TabM, with Fourier(t_x) WEAKLY concatenated to the input
     (decision 4 auxiliary path). `encode(reduce='mean')` -> z (B, d_block).
     [single representation; 32-submodel integration is a Phase-2 ablation]
  2. Memory:  P_k(t) = P_k^base + drift_k(Fourier(t))   (K prototypes)
  3. Retrieval: w_k = softmax(-||z - P_k(t_x)||^2 / tau)   [simple softmax, NO WTA]
  4. Aggregate: sum_k w_k * V_k,  V_k = label-dist embed + learned value
  5. Predict:  Predictor([z ; aggregated]) -> y_hat
  6. Loss:     L_main + lambda * L_smooth   (computed in the trainer / smoke test)

This is the memory+retrieval layer BETWEEN backbone and output -- not a head on
logits (EXPERIMENT_PLAN §1). WTA / TabR correction / multi-stage time injection
are intentionally absent (decision 3); they are Phase-2 ablations (§9).

Requires PyTorch + the tabm package (runs on the server).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .tabm_wrapper import TabMBackbone
from .temporal_embedding import FourierTimeEmbedding
from .prototype_memory import TimeIndexedPrototypeMemory
from .value_module import ValueModule
from .retrieval import MemoryRetrievalLayer


def _out_dim_for(task: str, n_classes: int) -> int:
    task = task.lower()
    if task == "binclass":
        return 2
    if task == "multiclass":
        return n_classes
    if task == "regression":
        return 1
    raise ValueError(f"Unknown task: {task}")


class Phase1Model(nn.Module):
    """TabM encoder + time-indexed prototype memory + softmax retrieval predictor.

    The representation/prototype dimension d equals the backbone d_block.
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int],
        task: str,
        n_classes: int = 2,
        # --- backbone (TabM) ---
        k: int = 32,
        n_blocks: int = 3,
        d_block: int = 512,
        dropout: float = 0.1,
        arch_type: str = "tabm",
        # --- memory / retrieval ---
        n_prototypes: int = 1000,
        rank: int = 32,
        mem_hidden: int = 64,
        tau_temp: float = 1.0,
        predictor_hidden: int = 256,
        predictor_mode: str = "concat",
        time_indexed: bool = True,
        # --- time embeddings ---
        time_periods: Sequence[float] = (1.0, 1.0 / 12, 1.0 / 52, 1.0 / 365),
        n_harmonics: int = 6,
        use_trend: bool = True,
        mem_time_out_dim: int = 32,
        # --- auxiliary input-time injection (decision 4) ---
        inject_time_input: bool = True,
        input_time_out_dim: int = 8,
    ):
        super().__init__()
        self.task = task.lower()
        self.d = d_block
        self.inject_time_input = inject_time_input
        out_dim = _out_dim_for(task, n_classes)

        # Auxiliary input-time embedding (weak): built first so we know its width.
        if inject_time_input:
            self.time_emb_input = FourierTimeEmbedding(
                out_dim=input_time_out_dim, periods=time_periods,
                n_harmonics=n_harmonics, use_trend=use_trend,
            )
            extra_num = self.time_emb_input.out_dim
        else:
            self.time_emb_input = None
            extra_num = 0

        # Backbone in representation mode (d_out=None) -> encode() gives z.
        self.backbone = TabMBackbone(
            n_num_features=n_num_features + extra_num,
            cat_cardinalities=cat_cardinalities,
            d_out=None,
            k=k, n_blocks=n_blocks, d_block=d_block,
            dropout=dropout, arch_type=arch_type,
        )

        # Memory time embedding (the PRIMARY time pathway) + prototype memory.
        self.time_emb_mem = FourierTimeEmbedding(
            out_dim=mem_time_out_dim, periods=time_periods,
            n_harmonics=n_harmonics, use_trend=use_trend,
        )
        self.memory = TimeIndexedPrototypeMemory(
            n_prototypes=n_prototypes, dim=d_block,
            time_embedding=self.time_emb_mem,
            rank=rank, hidden=mem_hidden, time_indexed=time_indexed,
        )
        self.value_module = ValueModule(
            n_prototypes=n_prototypes, dim=d_block, task=task, n_classes=n_classes,
        )
        self.retrieval = MemoryRetrievalLayer(
            dim=d_block, memory=self.memory, value_module=self.value_module,
            out_dim=out_dim, tau_temp=tau_temp, predictor_hidden=predictor_hidden,
            predictor_mode=predictor_mode,
        )

    def encode(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Backbone representation z (B, d), with weak input-time injection."""
        if self.inject_time_input:
            tau_in = self.time_emb_input(t)                      # (B, t_in)
            x_num = tau_in if x_num is None else torch.cat([x_num, tau_in], dim=1)
        return self.backbone.encode(x_num, x_cat, reduce="mean")  # (B, d)

    def forward(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
        t: torch.Tensor,
        return_aux: bool = False,
        ablate_memory: bool = False,
    ):
        """Returns y_hat (B, out_dim). If return_aux: (y_hat, {'w','sq_dist'}).

        ablate_memory=True zeroes the memory readout (diagnostics: mem contribution).
        """
        z = self.encode(x_num, x_cat, t)
        return self.retrieval(z, t, return_aux=return_aux, ablate_memory=ablate_memory)

    def smoothness_penalty(self, t: torch.Tensor) -> torch.Tensor:
        """L_smooth (0 when time_indexed=False)."""
        return self.retrieval.smoothness_penalty(t)

    @torch.no_grad()
    def init_memory_from_kmeans(self, base: np.ndarray) -> None:
        """Initialize P_base from time-sliced KMeans centroids in z-space.

        `base` is (K, d) centroids produced by the Phase-1 init utility
        (encode training data per time slice -> KMeans -> stack to K rows).
        See decision 5 / EXPERIMENT_PLAN §6.
        """
        self.memory.init_from_kmeans(base)
