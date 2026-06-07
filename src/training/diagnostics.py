"""Training-time diagnostics for the Phase-1 memory model.

We were tuning blind on final RMSE/AUC. These probe the INTERNALS each epoch so
fixes target the actual bottleneck. Headline number:

  mem_gap = loss(memory ablated) - loss(memory on)
      >0 and large  => the predictor actually USES the memory
      ~0            => memory is decorative (z-shortcut) -> the Test-1 null is
                       about the architecture, not the hypothesis

Plus: retrieval concentration (PR/top5), drift liveness (prototype path length),
predictor's weight mass on the memory branch vs z, and per-module gradient norms
(is the memory even getting learning signal?).

Requires PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _task_loss(out: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    if task == "regression":
        return F.mse_loss(out.squeeze(-1), y)
    return F.cross_entropy(out, y)


def grad_norms(model) -> dict:
    """L2 grad norm per component (call right AFTER loss.backward())."""
    b = {"backbone": 0.0, "proto_base": 0.0, "drift": 0.0,
         "value": 0.0, "predictor": 0.0, "other": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g2 = float(p.grad.detach().pow(2).sum().item())
        if name.startswith("backbone"):
            b["backbone"] += g2
        elif "memory.P_base" in name:
            b["proto_base"] += g2
        elif name.startswith("memory.time_mlp") or name == "memory.W":
            b["drift"] += g2
        elif name.startswith("value_module"):
            b["value"] += g2
        elif name.startswith("retrieval.predictor"):
            b["predictor"] += g2
        else:
            b["other"] += g2
    return {k: v ** 0.5 for k, v in b.items()}


def predictor_branch_split(model) -> dict:
    """Norm of the predictor's first-layer weights on the z half vs memory half."""
    lin = model.retrieval.predictor[0]            # Linear(2d, hidden)
    W = lin.weight.detach()                       # (hidden, 2d)
    d = W.shape[1] // 2
    wz = float(W[:, :d].norm().item())
    wagg = float(W[:, d:].norm().item())
    return {"w_z": wz, "w_agg": wagg, "agg_over_z": wagg / (wz + 1e-8)}


@torch.no_grad()
def forward_diagnostics(model, x_num, x_cat, t, y, task, *, n_times: int = 20) -> dict:
    """No-grad internal stats on a sample batch."""
    model.eval()
    out, aux = model(x_num, x_cat, t, return_aux=True)
    out0 = model(x_num, x_cat, t, ablate_memory=True)
    mem_gap = float(_task_loss(out0, y, task).item() - _task_loss(out, y, task).item())

    w = aux["w"]                                  # (B, K)
    K = int(w.shape[1])
    pr = float((1.0 / w.pow(2).sum(dim=1)).mean().item())
    top5 = float(w.topk(min(5, K), dim=1).values.sum(dim=1).mean().item())

    tg = torch.linspace(0.0, 1.0, n_times, device=w.device)
    P = model.memory.prototypes_at(tg)            # (T, K, d)
    path = (P[1:] - P[:-1]).norm(dim=2).sum(dim=0)   # (K,)

    agg_norm = float((w @ model.value_module.values()).norm(dim=1).mean().item())
    z = model.encode(x_num, x_cat, t)
    z_norm = float(z.norm(dim=1).mean().item())

    split = predictor_branch_split(model)
    return {
        "mem_gap": mem_gap,
        "PR_frac": pr / K, "top5_mass": top5,
        "path_mean": float(path.mean().item()), "path_max": float(path.max().item()),
        "agg_norm": agg_norm, "z_norm": z_norm, "agg_over_z_act": agg_norm / (z_norm + 1e-8),
        "w_agg_over_z": split["agg_over_z"],
    }


def format_line(epoch: int, fstats: dict, gnorms: dict) -> str:
    return (f"[diag e{epoch:03d}] mem_gap={fstats['mem_gap']:+.4f}  "
            f"PRfrac={fstats['PR_frac']:.3f} top5={fstats['top5_mass']:.2f}  "
            f"path={fstats['path_mean']:.3f}  "
            f"w_agg/z={fstats['w_agg_over_z']:.2f} act_agg/z={fstats['agg_over_z_act']:.2f}  "
            f"| grad bb={gnorms['backbone']:.1e} base={gnorms['proto_base']:.1e} "
            f"drift={gnorms['drift']:.1e} val={gnorms['value']:.1e} pred={gnorms['predictor']:.1e}")
