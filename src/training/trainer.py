"""Phase-0 training loop for the TabM baseline.

Follows the VERIFIED TabM training convention (official README):
  - TRAIN: optimize the MEAN LOSS over the k submodels (NOT the loss of the mean
    prediction). Output is (B, k, C)/(B, k, 1); loss is averaged over k.
  - INFER: average the k predictions; for classification average PROBABILITIES.
    (handled by TabMBackbone.predict)

Numeric features: TabReD saves RAW features; we quantile-normalize X_num (fit on
train), keep X_bin as 0/1, concat -> x_num. Regression targets are standardized
for training and de-standardized for the metric (Gorishniy/TabReD practice).

Requires PyTorch + sklearn + the tabm package (runs on the server, not here).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from tqdm.auto import tqdm

from ..data.tabred_loader import TabReDDataset, TabularSplit
from ..models.tabm_wrapper import TabMBackbone, compute_cat_cardinalities
from ..utils.metrics import compute_metric


@dataclass
class TrainConfig:
    k: int = 32
    n_blocks: int = 3
    d_block: int = 512
    dropout: float = 0.1
    arch_type: str = "tabm"
    lr: float = 2e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    patience: int = 16
    max_epochs: int = 200
    device: str = "cuda"
    seed_tag: str = ""               # shown in the tqdm progress bar (e.g. "s0")


def _prep_numeric(train: TabularSplit, *others: TabularSplit):
    """Quantile-normalize X_num (fit on train), keep X_bin as-is, concat -> x_num.

    Returns list of (concatenated float32 arrays) aligned with [train, *others],
    plus n_num_features (the concatenated width).
    """
    splits = [train, *others]
    has_num = train.X_num is not None
    has_bin = train.X_bin is not None

    qt = None
    if has_num:
        qt = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=max(min(train.X_num.shape[0] // 30, 1000), 10),
            subsample=10**9,
            random_state=0,
        ).fit(train.X_num)

    out = []
    for s in splits:
        parts = []
        if has_num:
            xn = qt.transform(s.X_num).astype(np.float32)
            # TabReD numeric features can contain NaN (missing values, e.g. sberbank).
            # QuantileTransformer passes NaN through -> would poison the model.
            # Impute to 0 (= the mean in the normal-output quantile space).
            xn = np.nan_to_num(xn, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(xn)
        if has_bin:
            xb = s.X_bin.astype(np.float32)
            xb = np.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(xb)
        x = np.concatenate(parts, axis=1) if parts else None
        out.append(x)
    n_num_features = 0 if out[0] is None else out[0].shape[1]
    return out, n_num_features


def _to_tensor(x: Optional[np.ndarray], dtype, device):
    return None if x is None else torch.as_tensor(x, dtype=dtype, device=device)


def train_tabm_baseline(data: TabReDDataset, cfg: TrainConfig) -> dict:
    """Train TabM on one dataset/seed and return {score, best_epoch, ...}."""
    device = cfg.device if torch.cuda.is_available() else "cpu"
    task = data.task

    # ---- features ----
    (xnum_tr, xnum_va, xnum_te), n_num = _prep_numeric(data.train, data.val, data.test)
    cat_card = compute_cat_cardinalities(data.train.X_cat)

    x_num = {p: _to_tensor(a, torch.float32, device)
             for p, a in zip(("train", "val", "test"), (xnum_tr, xnum_va, xnum_te))}
    x_cat = {p: _to_tensor(getattr(data, p).X_cat, torch.long, device)
             for p in ("train", "val", "test")}

    # ---- targets ----
    y_np = {p: getattr(data, p).y for p in ("train", "val", "test")}
    if task == "regression":
        y_mean = float(np.mean(y_np["train"])); y_std = float(np.std(y_np["train"]) + 1e-8)
        y = {p: torch.as_tensor((y_np[p] - y_mean) / y_std, dtype=torch.float32, device=device)
             for p in y_np}
        d_out = 1
    else:
        n_classes = int(max(int(y_np["train"].max()), int(y_np["val"].max())) + 1)
        y = {p: torch.as_tensor(y_np[p], dtype=torch.long, device=device) for p in y_np}
        d_out = n_classes if task == "multiclass" else 2

    # ---- model ----
    model = TabMBackbone(
        n_num_features=n_num, cat_cardinalities=cat_card, d_out=d_out,
        k=cfg.k, n_blocks=cfg.n_blocks, d_block=cfg.d_block,
        dropout=cfg.dropout, arch_type=cfg.arch_type,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_train = (x_num["train"] if x_num["train"] is not None else x_cat["train"]).shape[0]

    def batches():
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, cfg.batch_size):
            yield perm[i:i + cfg.batch_size]

    def mean_loss(out, target):
        # out: (B, k, C) or (B, k, 1)
        B, k = out.shape[0], out.shape[1]
        if task == "regression":
            tgt = target.view(B, 1, 1).expand(B, k, 1)
            return F.mse_loss(out, tgt)
        return F.cross_entropy(out.reshape(B * k, -1), target.repeat_interleave(k))

    @torch.no_grad()
    def evaluate(part: str) -> float:
        model.eval()
        pred = model.predict(x_num[part], x_cat[part], task).cpu().numpy()
        if task == "regression":
            pred = pred * y_std + y_mean
            return compute_metric(y_np[part], pred, task)
        if task == "binclass":
            return compute_metric(y_np[part], pred[:, 1], task)  # P(positive)
        return compute_metric(y_np[part], pred, task)

    higher_better = task != "regression"
    best = -np.inf if higher_better else np.inf
    best_epoch, since_improve, best_state = -1, 0, None

    pbar = tqdm(
        range(cfg.max_epochs),
        desc=f"{data.name}[{getattr(cfg, 'seed_tag', '')}]",
        leave=False, dynamic_ncols=True,
    )
    for epoch in pbar:
        model.train()
        for idx in batches():
            xb_num = None if x_num["train"] is None else x_num["train"][idx]
            xb_cat = None if x_cat["train"] is None else x_cat["train"][idx]
            out = model(xb_num, xb_cat)               # (B, k, d_out)
            loss = mean_loss(out, y["train"][idx])
            if not torch.isfinite(loss):
                # Diverged (NaN/inf loss). Skip this step rather than poisoning weights.
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # stabilize
            opt.step()

        val = evaluate("val")
        improved = (val > best) if higher_better else (val < best)
        pbar.set_postfix(val=f"{val:.4f}", best=f"{best if np.isfinite(best) else val:.4f}",
                         patience=f"{since_improve}/{cfg.patience}")
        if improved:
            best, best_epoch, since_improve = val, epoch, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            since_improve += 1
            if since_improve > cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_score = evaluate("test")
    return {
        "dataset": data.name, "task": task, "split": data.split,
        "val_score": float(best), "score": float(test_score),
        "best_epoch": int(best_epoch),
    }
