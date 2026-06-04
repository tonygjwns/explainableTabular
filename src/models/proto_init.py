"""Time-sliced KMeans initialization for the prototype memory (decision 5).

EXPERIMENT_PLAN §6 / decision 5: instead of random init, split the training data
into time slices, run KMeans within each slice (in the backbone's representation
space z), and stack the centroids into the K prototype bases P_base. The drift
function then only has to fine-tune, which converges faster and more stably.

Prototypes are compared to z in retrieval (||z - P_k(t)||), so the centroids must
live in the SAME z-space -- we encode through `model.encode` (which includes the
weak input-time injection), NOT on raw features.

Usage (in the Phase-1 trainer, before training):
    base = time_sliced_kmeans_init(model, Xn_tr, Xc_tr, t_tr, K=cfg.K)
    model.init_memory_from_kmeans(base)

Requires PyTorch + scikit-learn.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.cluster import KMeans


def _even_counts(K: int, n_slices: int) -> list[int]:
    """Split K cluster budget as evenly as possible across n_slices."""
    base = K // n_slices
    rem = K - base * n_slices
    return [base + (1 if i < rem else 0) for i in range(n_slices)]


@torch.no_grad()
def _encode_all(model, x_num, x_cat, t, *, device, batch: int) -> np.ndarray:
    """Encode rows through model.encode in batches -> (N, d) numpy (z-space)."""
    model.eval()
    N = (x_num if x_num is not None else x_cat).shape[0]
    t_t = torch.as_tensor(t, dtype=torch.float32, device=device)
    xn = None if x_num is None else torch.as_tensor(x_num, dtype=torch.float32, device=device)
    xc = None if x_cat is None else torch.as_tensor(x_cat, dtype=torch.long, device=device)
    out = []
    for i in range(0, N, batch):
        sl = slice(i, i + batch)
        zb = model.encode(None if xn is None else xn[sl],
                          None if xc is None else xc[sl],
                          t_t[sl])
        out.append(zb.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def time_sliced_kmeans_init(
    model,
    x_num: Optional[np.ndarray],
    x_cat: Optional[np.ndarray],
    t: np.ndarray,
    *,
    K: int,
    n_slices: int = 10,
    max_samples: int = 50_000,
    encode_batch: int = 8192,
    device: Optional[str] = None,
    seed: int = 0,
) -> np.ndarray:
    """Return (K, d) prototype-base centroids from time-sliced KMeans in z-space.

    Args:
        model: a Phase1Model (uses model.encode and model.d).
        x_num, x_cat: TRAIN feature arrays (numpy) or None.
        t: TRAIN normalized timestamps (numpy), shape (N,).
        K: number of prototypes (centroids in total).
        n_slices: number of equal-count time slices.
        max_samples: subsample this many train rows before encoding (cost guard
            for million-row datasets; KMeans on millions is needless here).
        encode_batch: batch size for the encoding pass.
        device: where to encode ('cuda'/'cpu'); defaults to model's device.
        seed: RNG seed (subsample + KMeans).
    Returns:
        base: (K, d) float32 array, ready for model.init_memory_from_kmeans(base).
    """
    rng = np.random.default_rng(seed)
    if device is None:
        device = next(model.parameters()).device.type

    N = (x_num if x_num is not None else x_cat).shape[0]
    t = np.asarray(t).reshape(-1)

    # subsample rows (cost guard) -- keep them, slice by time afterwards
    if N > max_samples:
        sel = rng.choice(N, size=max_samples, replace=False)
        x_num = None if x_num is None else x_num[sel]
        x_cat = None if x_cat is None else x_cat[sel]
        t = t[sel]
        N = max_samples

    Z = _encode_all(model, x_num, x_cat, t, device=device, batch=encode_batch)  # (N, d)
    d = Z.shape[1]

    # equal-count time slices via the time-sorted order
    order = np.argsort(t, kind="stable")
    slice_idx = np.array_split(order, n_slices)
    targets = _even_counts(K, n_slices)

    centroids = []
    for idx_in_slice, n_target in zip(slice_idx, targets):
        if n_target <= 0 or len(idx_in_slice) == 0:
            continue
        Zs = Z[idx_in_slice]
        n_clusters = int(min(n_target, len(Zs)))
        if n_clusters <= 0:
            continue
        if n_clusters == len(Zs):
            centroids.append(Zs.astype(np.float32))   # each point is its own center
            continue
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=seed)
        km.fit(Zs)
        centroids.append(km.cluster_centers_.astype(np.float32))

    base = np.concatenate(centroids, axis=0) if centroids else np.zeros((0, d), np.float32)

    # pad/trim to exactly K rows
    if base.shape[0] < K:
        deficit = K - base.shape[0]
        fill = Z[rng.choice(Z.shape[0], size=deficit, replace=True)].astype(np.float32)
        base = np.concatenate([base, fill], axis=0)
    elif base.shape[0] > K:
        base = base[:K]

    return np.ascontiguousarray(base, dtype=np.float32)
