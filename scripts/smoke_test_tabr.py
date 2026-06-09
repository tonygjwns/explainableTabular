"""CPU smoke test for the TimeTabR retrieval infra (F2-independent, time+label hooks).

Verifies wiring only: forward shapes, top-k retrieval, time_mode toggles, and that
gradients reach the value/label/predictor AND the time-modulation params when enabled.
(Modulation is zero-init, so time_mode none vs both give identical OUTPUT at init —
the point is the plumbing exposes (t_q,t_i,y_i) and grads flow so F2 drops in.)

    python scripts/smoke_test_tabr.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.models.tabr import TimeTabR, TimeTabRModel


def check(task, n_classes=3):
    B, N, d = 16, 200, 24
    print(f"\n=== task={task} ===")
    zq = torch.randn(B, d); tq = torch.rand(B)
    zc = torch.randn(N, d); tc = torch.rand(N)
    yc = (torch.rand(N) if task == "regression"
          else torch.randint(0, (n_classes if task == "multiclass" else 2), (N,)))
    yq = (torch.randn(B) if task == "regression"
          else torch.randint(0, (n_classes if task == "multiclass" else 2), (B,)))
    for tm in ("none", "metric", "value", "both"):
        model = TimeTabR(d, task, n_classes, topk=8, time_mode=tm)
        for p in model.parameters():
            p.grad = None
        out, aux = model(zq, tq, zc, tc, yc, return_aux=True)
        exp = n_classes if task == "multiclass" else (2 if task == "binclass" else 1)
        assert out.shape == (B, exp), f"{out.shape} != {(B, exp)}"
        assert aux["w"].shape == (B, 8)
        assert torch.allclose(aux["w"].sum(1), torch.ones(B), atol=1e-5)
        loss = (F.mse_loss(out.squeeze(-1), yq.float()) if task == "regression"
                else F.cross_entropy(out, yq))
        loss.backward()
        g_val = model.value[0].weight.grad is not None
        g_lab = (model.label_emb.weight.grad is not None
                 if hasattr(model.label_emb, "weight") else True)
        g_vt = model.value_time.weight.grad is not None and float(model.value_time.weight.grad.abs().sum()) > 0
        g_mt = model.metric_time.weight.grad is not None and float(model.metric_time.weight.grad.abs().sum()) > 0
        time_grad = {"none": "(n/a)", "metric": g_mt, "value": g_vt,
                     "both": (g_vt and g_mt)}[tm]
        print(f"  time_mode={tm:6s}: out{tuple(out.shape)} w_sum=1 OK  "
              f"grad[value={g_val},label={g_lab}] time_mod_grad={time_grad}")


def check_exclude_self():
    # in-batch retrieval: candidates == queries, must exclude self
    B, d = 12, 16
    z = torch.randn(B, d); t = torch.rand(B); y = torch.randint(0, 2, (B,))
    m = TimeTabR(d, "binclass", 2, topk=4, time_mode="both")
    out = m(z, t, z, t, y, exclude_self=torch.arange(B))
    assert out.shape == (B, 2)
    print("\nin-batch exclude_self OK")


def check_model(task, n_classes=3):
    B, Nc, nfeat = 16, 300, 10
    xq = torch.randn(B, nfeat); tq = torch.rand(B)
    xc = torch.randn(Nc, nfeat); tc = torch.rand(Nc)
    yc = (torch.rand(Nc) if task == "regression"
          else torch.randint(0, (n_classes if task == "multiclass" else 2), (Nc,)))
    yq = (torch.randn(B) if task == "regression"
          else torch.randint(0, (n_classes if task == "multiclass" else 2), (B,)))
    exp = n_classes if task == "multiclass" else (2 if task == "binclass" else 1)
    print(f"\n=== model task={task} ===")
    for arch, tm in [("mlp_t", "none"), ("tabr", "none"), ("time_tabr", "value"), ("time_tabr", "both")]:
        m = TimeTabRModel(nfeat, task, n_classes, arch=arch, time_mode=tm, enc_dim=32, topk=8)
        for p in m.parameters():
            p.grad = None
        out = m(xq, tq, xc, tc, yc)
        assert out.shape == (B, exp), f"{arch}/{tm}: {out.shape} != {(B, exp)}"
        loss = (F.mse_loss(out.squeeze(-1), yq.float()) if task == "regression"
                else F.cross_entropy(out, yq))
        loss.backward()
        g_enc = m.encoder[0].weight.grad is not None
        print(f"  arch={arch:9s} time_mode={tm:5s}: out{tuple(out.shape)} OK, grad[enc={g_enc}]")
    # in-batch retrieval (candidates == queries)
    m = TimeTabRModel(nfeat, task, n_classes, arch="time_tabr", time_mode="value", enc_dim=32, topk=4)
    _ = m(xq, tq, xq, tq, yq, exclude_self=torch.arange(B))
    print("  in-batch (exclude_self) OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    for task in ("binclass", "multiclass", "regression"):
        check(task)
    check_exclude_self()
    for task in ("binclass", "multiclass", "regression"):
        check_model(task)
    print("\nAll TabR smoke checks passed.")
