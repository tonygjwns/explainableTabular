"""Test 3 analysis: retrieval concentration + prototype-trajectory geometry.

PRE_REGISTRATION §3.3 / EXPERIMENT_PLAN §8 Test 3 (interpretability, non-gating
qualitative PASS by peer+self). This module provides the QUANTITATIVE aids and the
trajectory plot that the human judgment is made on:

  retrieval_concentration(model, X..., t)  -> are inputs retrieved to a FEW
      prototypes (not uniform)?  participation ratio, entropy, top-k mass.
  prototype_trajectories(model)            -> P_k(t) on a time grid -> (T, K, d)
  trajectory_metrics(P)                    -> path length, net displacement,
      straightness (=net/path; 1=straight line, ->0=random walk) per prototype
  plot_trajectories(...)                   -> 2D PCA/UMAP plot, colored by t

Requires PyTorch (+ sklearn for PCA; matplotlib only inside plot fn; umap optional).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


@torch.no_grad()
def retrieval_concentration(
    model,
    x_num: Optional[np.ndarray],
    x_cat: Optional[np.ndarray],
    t: np.ndarray,
    *,
    device: Optional[str] = None,
    batch: int = 1024,
    max_samples: int = 20_000,
    seed: int = 0,
) -> dict:
    """Concentration of retrieval weights w (B,K) over a sample of inputs.

    Returned fracs are normalized so UNIFORM ~ 1 and concentrated ~ 0:
      participation_ratio_frac = (1/sum w^2)/K
      entropy_frac             = H(w)/log K
    plus absolute top-1 / top-5 retrieved mass.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device.type
    N = (x_num if x_num is not None else x_cat).shape[0]
    rng = np.random.default_rng(seed)
    if N > max_samples:
        sel = rng.choice(N, size=max_samples, replace=False)
        x_num = None if x_num is None else x_num[sel]
        x_cat = None if x_cat is None else x_cat[sel]
        t = np.asarray(t)[sel]
        N = max_samples

    xn = None if x_num is None else torch.as_tensor(x_num, dtype=torch.float32, device=device)
    xc = None if x_cat is None else torch.as_tensor(x_cat, dtype=torch.long, device=device)
    tt = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=device)
    K = int(model.memory.K)

    pr_sum = ent_sum = top1_sum = top5_sum = 0.0
    n = 0
    for i in range(0, N, batch):
        sl = slice(i, i + batch)
        _, aux = model(None if xn is None else xn[sl],
                       None if xc is None else xc[sl],
                       tt[sl], return_aux=True)
        w = aux["w"]                                   # (b, K)
        pr_sum += float((1.0 / w.pow(2).sum(dim=1)).sum())
        ent_sum += float((-(w.clamp_min(1e-12).log() * w).sum(dim=1)).sum())
        top = w.topk(min(5, K), dim=1).values
        top1_sum += float(top[:, 0].sum())
        top5_sum += float(top.sum())
        n += int(w.shape[0])

    logK = float(np.log(K))
    return {
        "K": K, "n": n,
        "participation_ratio_mean": pr_sum / n,
        "participation_ratio_frac": (pr_sum / n) / K,
        "entropy_mean": ent_sum / n,
        "entropy_max": logK,
        "entropy_frac": (ent_sum / n) / logK if logK > 0 else 0.0,
        "top1_mass_mean": top1_sum / n,
        "top5_mass_mean": top5_sum / n,
    }


@torch.no_grad()
def prototype_trajectories(
    model, *, n_times: int = 50, t_min: float = 0.0, t_max: float = 1.0,
    device: Optional[str] = None,
):
    """Return (t_grid (T,), P (T, K, d)) by evaluating P_k(t) on a time grid."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device.type
    tg = torch.linspace(float(t_min), float(t_max), int(n_times), device=device)
    P = model.memory.prototypes_at(tg)                 # (T, K, d)
    return tg.cpu().numpy(), P.cpu().numpy()


def trajectory_metrics(P: np.ndarray) -> dict:
    """Geometry of each prototype's path. P: (T, K, d).

    straightness = ||P_T - P_0|| / sum_t ||P_{t+1}-P_t||  (1=straight, ->0=wiggly).
    """
    P = np.asarray(P)
    steps = np.linalg.norm(np.diff(P, axis=0), axis=2)   # (T-1, K)
    path_len = steps.sum(axis=0)                          # (K,)
    net = np.linalg.norm(P[-1] - P[0], axis=1)            # (K,)
    straight = np.where(path_len > 1e-12, net / np.maximum(path_len, 1e-12), 0.0)
    movers = np.argsort(-path_len)
    return {
        "path_len_mean": float(path_len.mean()),
        "path_len_median": float(np.median(path_len)),
        "net_disp_mean": float(net.mean()),
        "straightness_mean": float(straight.mean()),
        "straightness_median": float(np.median(straight)),
        "movers_idx": movers[:20].tolist(),
        "_path_len": path_len, "_straightness": straight, "_net": net,
    }


def plot_trajectories(
    P: np.ndarray, t_grid: np.ndarray, out_png: str, *,
    n_proto: int = 20, mover_idx: Optional[list] = None, method: str = "pca",
) -> str:
    """2D projection of selected prototype trajectories, colored by t. Saves PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    P = np.asarray(P)
    T, K, d = P.shape
    flat = P.reshape(T * K, d)
    if method == "umap":
        try:
            import umap  # type: ignore
            emb = umap.UMAP(n_components=2, random_state=0).fit_transform(flat)
        except Exception:
            method = "pca"
            emb = PCA(n_components=2, random_state=0).fit_transform(flat)
    else:
        emb = PCA(n_components=2, random_state=0).fit_transform(flat)
    emb = emb.reshape(T, K, 2)

    idx = (mover_idx if mover_idx is not None else list(range(K)))[:n_proto]
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = None
    for k in idx:
        ax.plot(emb[:, k, 0], emb[:, k, 1], "-", alpha=0.35, lw=0.8, color="gray")
        sc = ax.scatter(emb[:, k, 0], emb[:, k, 1], c=t_grid, cmap="viridis", s=10)
    if sc is not None:
        fig.colorbar(sc, ax=ax, label="t (normalized; train range -> [0,1])")
    ax.set_title(f"Prototype trajectories P_k(t) — {method.upper()}, top-{len(idx)} movers")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
