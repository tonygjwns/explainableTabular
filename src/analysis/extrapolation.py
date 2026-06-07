"""Test 2 analysis: does the prototype memory EXTRAPOLATE to track real drift?

PRE_REGISTRATION §3.2 / EXPERIMENT_PLAN §8 Test 2. Train on the first 70% of
training time, then for held-out FUTURE time slices compare:
  - C_extrap(s): the memory's usage-weighted prototype centroid at time t_s,
    extrapolated (t_s is beyond the trained time range),
        C_extrap(s) = sum_k wbar_k * P_k(t_s),  wbar = mean retrieval weight on train
  - C_real(s):   the mean backbone embedding of the REAL data in slice s.

If the drift learned smooth structure that generalizes, C_extrap should track the
real distribution shift C_real across the future slices.

Metrics (PRE_REG mentions R^2>=30% and, for H1b, Pearson r>=0.4):
  - variance-weighted mean Pearson r over dims (weight = real movement variance)
  - variance-weighted mean per-dim R^2 of predicting C_real from C_extrap
  - direction agreement: do net (last-first) shifts share sign (|real|-weighted)?
A fixed memory (no drift) gives C_extrap constant over s => r ~ 0 by construction,
so a clearly positive r is evidence the drift extrapolation is meaningful.

NOTE: with only S~8 future slices these metrics are noisy; report alongside the
trajectory plot and treat as evidence, not a hard pass by itself.

Requires PyTorch + numpy.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


@torch.no_grad()
def _encode_all(model, x_num, x_cat, t, *, device, batch=1024) -> np.ndarray:
    model.eval()
    N = (x_num if x_num is not None else x_cat).shape[0]
    tt = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=device)
    xn = None if x_num is None else torch.as_tensor(x_num, dtype=torch.float32, device=device)
    xc = None if x_cat is None else torch.as_tensor(x_cat, dtype=torch.long, device=device)
    out = []
    for i in range(0, N, batch):
        sl = slice(i, i + batch)
        out.append(model.encode(None if xn is None else xn[sl],
                                None if xc is None else xc[sl], tt[sl]).cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def mean_retrieval_weights(model, x_num, x_cat, t, *, device, batch=1024,
                           max_samples=20_000, seed=0) -> np.ndarray:
    """Average retrieval weight per prototype over (a sample of) data: (K,)."""
    model.eval()
    N = (x_num if x_num is not None else x_cat).shape[0]
    rng = np.random.default_rng(seed)
    if N > max_samples:
        sel = rng.choice(N, size=max_samples, replace=False)
        x_num = None if x_num is None else x_num[sel]
        x_cat = None if x_cat is None else x_cat[sel]
        t = np.asarray(t)[sel]; N = max_samples
    tt = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=device)
    xn = None if x_num is None else torch.as_tensor(x_num, dtype=torch.float32, device=device)
    xc = None if x_cat is None else torch.as_tensor(x_cat, dtype=torch.long, device=device)
    acc = torch.zeros(int(model.memory.K), device=device)
    n = 0
    for i in range(0, N, batch):
        sl = slice(i, i + batch)
        _, aux = model(None if xn is None else xn[sl],
                       None if xc is None else xc[sl], tt[sl], return_aux=True)
        acc += aux["w"].sum(dim=0)
        n += int(aux["w"].shape[0])
    return (acc / max(n, 1)).cpu().numpy()


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def extrapolation_fit(
    model, late_x_num, late_x_cat, late_t, wbar: np.ndarray,
    *, n_slices: int = 8, device: Optional[str] = None, batch: int = 1024,
) -> dict:
    """Compare extrapolated memory centroid vs real future-data centroid per slice."""
    if device is None:
        device = next(model.parameters()).device.type
    z = _encode_all(model, late_x_num, late_x_cat, late_t, device=device, batch=batch)  # (N,d)
    t = np.asarray(late_t).reshape(-1)
    order = np.argsort(t, kind="stable")
    slices = np.array_split(order, n_slices)

    C_real, t_s = [], []
    for idx in slices:
        if len(idx) == 0:
            continue
        C_real.append(z[idx].mean(axis=0))
        t_s.append(float(t[idx].mean()))
    C_real = np.asarray(C_real)                     # (S, d)
    t_s = np.asarray(t_s)                           # (S,)
    S, d = C_real.shape

    P = model.memory.prototypes_at(torch.as_tensor(t_s, dtype=torch.float32, device=device))
    P = P.cpu().numpy()                             # (S, K, d)
    w = np.asarray(wbar, float)
    C_extrap = np.einsum("skd,k->sd", P, w)         # (S, d)

    var_real = C_real.var(axis=0)                   # (d,)
    wsum = float(var_real.sum()) + 1e-12

    # variance-weighted Pearson r and per-dim R^2
    r_d = np.array([_pearson(C_extrap[:, j], C_real[:, j]) for j in range(d)])
    mean_r = float((r_d * var_real).sum() / wsum)
    r2_d = np.clip(r_d, 0, None) ** 2               # R^2 from correlation (>=0)
    mean_r2 = float((r2_d * var_real).sum() / wsum)

    # direction agreement of net shift (first->last slice), |real|-weighted
    net_real = C_real[-1] - C_real[0]
    net_extr = C_extrap[-1] - C_extrap[0]
    agree = (np.sign(net_real) == np.sign(net_extr)).astype(float)
    dir_agree = float((agree * np.abs(net_real)).sum() / (np.abs(net_real).sum() + 1e-12))

    return {
        "n_slices": int(S),
        "mean_pearson_r_weighted": mean_r,
        "mean_r2_weighted": mean_r2,
        "direction_agreement_frac": dir_agree,
        "real_centroid_movement": float(np.linalg.norm(net_real)),
        "extrap_centroid_movement": float(np.linalg.norm(net_extr)),
    }
