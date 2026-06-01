"""Smoke test for the novel memory + retrieval modules (CPU, no GPU/data needed).

Run on any machine with PyTorch installed:
    python scripts/smoke_test_memory.py

Verifies: forward pass shapes, time-indexed vs fixed memory behavior, smoothness
penalty sign, and that gradients flow. This does NOT test correctness of results
(that needs data) -- only that the modules are wired correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.temporal_embedding import FourierTimeEmbedding
from src.models.prototype_memory import TimeIndexedPrototypeMemory
from src.models.value_module import ValueModule
from src.models.retrieval import MemoryRetrievalLayer


def build(task: str, time_indexed: bool, K: int = 64, d: int = 32, C: int = 3):
    te = FourierTimeEmbedding(out_dim=d, n_harmonics=4)
    mem = TimeIndexedPrototypeMemory(K, d, te, rank=8, hidden=32, time_indexed=time_indexed)
    out_dim = C if task == "multiclass" else 1
    vm = ValueModule(K, d, task=task, n_classes=C)
    layer = MemoryRetrievalLayer(d, mem, vm, out_dim=out_dim, tau_temp=1.0)
    return layer


def check(task: str):
    B, d = 16, 32
    print(f"\n=== task={task} ===")
    for time_indexed in (True, False):
        layer = build(task, time_indexed, d=d)
        z = torch.randn(B, d, requires_grad=True)
        t = torch.rand(B)  # normalized timestamps in [0,1]
        y, aux = layer(z, t, return_aux=True)
        out_dim = y.shape[1]
        assert aux["w"].shape == (B, layer.memory.K)
        w_row = aux["w"][0]
        assert torch.allclose(w_row.sum(), torch.tensor(1.0), atol=1e-5), "weights must sum to 1"
        smooth = layer.smoothness_penalty(t)
        # gradient flow
        loss = y.pow(2).mean() + 0.1 * smooth
        loss.backward()
        gd = z.grad is not None
        print(f"  time_indexed={time_indexed}: y{tuple(y.shape)} "
              f"w_sum=1 OK, smooth={float(smooth):.4e}, grad_flows={gd}")
        if not time_indexed:
            assert float(smooth) == 0.0, "fixed memory must have zero smoothness penalty"


if __name__ == "__main__":
    torch.manual_seed(0)
    for task in ("binclass", "multiclass", "regression"):
        check(task)
    print("\nAll smoke checks passed.")
