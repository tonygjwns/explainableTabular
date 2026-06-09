"""Q1 functional faithfulness (PLAN_RESCUE §A) — gauge-fixed recovery of w(t).

The MAIN claim's falsifiable test. The model's effective decision direction at time t
is the input gradient of its decision score:
    w_hat(t) = E_x[ ∂ decision(x, t) / ∂x ].
Recovery = per-t cosine to the TRUE drift direction:
    recovery(t) = cos( w_hat(t), w_true(t) ).
w_hat and w_true live in the SAME input space → the gauge is already fixed →
**NO Procrustes/CCA** (a free rotation would inflate alignment = false PASS). The
synthetic must have its decision LINEAR in x so w_hat(t) is x-stable and unambiguous.

Requires PyTorch.
"""
from __future__ import annotations

import numpy as np
import torch


def effective_weight(score_fn, x_np: np.ndarray, t_val: float, device: str) -> np.ndarray:
    """w_hat(t) = mean_x ∂ score_fn(x,t)/∂x  for a fixed t. Returns (d,) numpy."""
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device).requires_grad_(True)
    t = torch.full((x.shape[0],), float(t_val), dtype=torch.float32, device=device)
    s = score_fn(x, t)                                  # (M,) decision score (logit diff)
    g, = torch.autograd.grad(s.sum(), x)                # (M, d)
    return g.detach().mean(dim=0).cpu().numpy()


def recovery_curve(score_fn, w_true_fn, t_grid, x_np: np.ndarray, device: str):
    """recovery(t)=cos(w_hat(t), w_true(t)) over the grid. Returns (rec[T], w_hat[T,d])."""
    rec, whs = [], []
    for tv in t_grid:
        wh = effective_weight(score_fn, x_np, float(tv), device)
        wt = np.asarray(w_true_fn(float(tv)), dtype=np.float64)
        denom = (np.linalg.norm(wh) * np.linalg.norm(wt)) + 1e-12
        rec.append(float(np.dot(wh, wt) / denom))
        whs.append(wh)
    return np.asarray(rec), np.asarray(whs)


def band_line(lo: float, hi: float, frac: float) -> float:
    """PASS/FAIL line at a fraction of the (floor, ceiling) range."""
    return lo + frac * (hi - lo)
