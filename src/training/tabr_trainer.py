"""Training loop for the 3-arm TimeTabRModel (Q2b structure axis).

Trains TimeTabRModel (shared MLP encoder; arch in {mlp_t, tabr, time_tabr}) on one
dataset/seed with a single supervised loss (CE / MSE). This is the runner-side half
of the Q2b infra; the model + retrieval core live in src/models/tabr.py.

Retrieval protocol (NEXT_TAB.md ★):
- TRAIN: in-batch retrieval. Each minibatch is BOTH the queries and the candidate
  context (context = same batch), with exclude_self=arange(B) so a query never
  retrieves itself. No load-balance / L_smooth term — top-k retrieval doesn't
  collapse the way a softmax-over-K prototype memory does.
- EVAL : a FIXED context set sampled once from train (default 4096 instances) is the
  candidate pool for every query. The encoder is re-applied to that context at each
  evaluate() call (weights change across epochs), no_grad.

arch='mlp_t' ignores the context entirely (time is just a feature) — so the same
loop trains all three arms; the factorial runner only flips cfg.arch / cfg.time_mode.

Reuses the Phase-0 numeric prep / global cat-cardinality helpers; categoricals are
appended as one-hot so the encoder sees a single flat numeric matrix.
Requires PyTorch + sklearn (runs on the server).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..data.tabred_loader import TabReDDataset
from ..models.tabr import TimeTabRModel
from ..utils.metrics import compute_metric
from .trainer import _prep_numeric, _to_tensor, _global_cat_cardinalities


@dataclass
class TabRConfig:
    # architecture (the Q2b factorial axes)
    arch: str = "time_tabr"          # mlp_t | tabr | time_tabr
    time_mode: str = "value"         # value | metric | both (only used when arch=time_tabr)
    time_basis: str = "trend"        # trend (extrapolation-safe) | fourier
    trend_degree: int = 3
    time_out: int = 16
    # shared encoder
    enc_dim: int = 128
    enc_hidden: int = 256
    n_enc_layers: int = 2
    # retrieval
    topk: int = 32
    predictor_hidden: int = 256
    eval_context_size: int = 4096    # fixed candidate pool sampled from train
    # optimization
    lr: float = 2e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    eval_batch: int = 1024
    patience: int = 16
    min_epochs: int = 0          # floor before early-stop may fire (give zero-init
                                 # time-modulation room to train; 0 = off)
    max_epochs: int = 200
    device: str = "cuda"
    seed_tag: str = ""
    record_history: bool = False  # collect per-epoch val into the result (diagnostic)


def _build_features(data: TabReDDataset):
    """Prepared numeric (quantile X_num + X_bin) with categoricals appended one-hot.

    The encoder takes a single flat float matrix, so categoricals are one-hot
    encoded with GLOBAL per-column cardinalities (a temporal test split can hold
    codes unseen in train). Returns ({part: float32 ndarray}, n_features).
    """
    (xnum_tr, xnum_va, xnum_te), _ = _prep_numeric(data.train, data.val, data.test)
    num = {"train": xnum_tr, "val": xnum_va, "test": xnum_te}
    cat_card = _global_cat_cardinalities(data.train.X_cat, data.val.X_cat, data.test.X_cat)

    feats = {}
    for p in ("train", "val", "test"):
        parts = []
        if num[p] is not None:
            parts.append(num[p])
        xcat = getattr(data, p).X_cat
        if cat_card and xcat is not None and xcat.size:
            oh = []
            for j, card in enumerate(cat_card):
                col = np.clip(xcat[:, j].astype(np.int64), 0, card - 1)
                e = np.zeros((xcat.shape[0], card), dtype=np.float32)
                e[np.arange(xcat.shape[0]), col] = 1.0
                oh.append(e)
            parts.append(np.concatenate(oh, axis=1))
        feats[p] = (np.concatenate(parts, axis=1).astype(np.float32)
                    if parts else np.zeros((getattr(data, p).y.shape[0], 0), dtype=np.float32))
    n_features = feats["train"].shape[1]
    return feats, n_features


def train_timetabr(data: TabReDDataset, cfg: TabRConfig) -> dict:
    """Train one TimeTabRModel arm on one dataset/seed.

    Returns {dataset, task, split, arch, time_mode, val_score, score, best_epoch, model}.
    """
    device = cfg.device if torch.cuda.is_available() else "cpu"
    task = data.task

    feats, n_features = _build_features(data)
    x = {p: _to_tensor(feats[p], torch.float32, device) for p in feats}
    t = {p: _to_tensor(getattr(data, p).t, torch.float32, device)
         for p in ("train", "val", "test")}

    # ---- targets ----
    y_np = {p: getattr(data, p).y for p in ("train", "val", "test")}
    if task == "regression":
        y_mean = float(np.mean(y_np["train"])); y_std = float(np.std(y_np["train"]) + 1e-8)
        y = {p: torch.as_tensor((y_np[p] - y_mean) / y_std, dtype=torch.float32, device=device)
             for p in y_np}
        n_classes = 2  # unused
    else:
        n_classes = int(max(int(y_np["train"].max()), int(y_np["val"].max()),
                            int(y_np["test"].max())) + 1)
        y = {p: torch.as_tensor(y_np[p], dtype=torch.long, device=device) for p in y_np}

    model = TimeTabRModel(
        n_features, task, n_classes, arch=cfg.arch, time_mode=cfg.time_mode,
        enc_dim=cfg.enc_dim, enc_hidden=cfg.enc_hidden, n_enc_layers=cfg.n_enc_layers,
        time_basis=cfg.time_basis, trend_degree=cfg.trend_degree, time_out=cfg.time_out,
        topk=cfg.topk, predictor_hidden=cfg.predictor_hidden,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_train = x["train"].shape[0]
    needs_ctx = cfg.arch != "mlp_t"

    # fixed eval context = a sample of train instances (reused every evaluate())
    if needs_ctx:
        n_ctx = min(cfg.eval_context_size, n_train)
        ctx_idx = torch.randperm(n_train, device=device)[:n_ctx]
        ctx_x_eval, ctx_t_eval, ctx_y_eval = x["train"][ctx_idx], t["train"][ctx_idx], y["train"][ctx_idx]

    def batches():
        perm = torch.randperm(n_train, device=device)
        for i in range(0, n_train, cfg.batch_size):
            yield perm[i:i + cfg.batch_size]

    def main_loss(y_hat, target):
        if task == "regression":
            return F.mse_loss(y_hat.squeeze(-1), target)
        return F.cross_entropy(y_hat, target)

    @torch.no_grad()
    def evaluate(part: str) -> float:
        model.eval()
        n = x[part].shape[0]
        preds = []
        for i in range(0, n, cfg.eval_batch):
            sl = slice(i, i + cfg.eval_batch)
            if needs_ctx:
                out = model(x[part][sl], t[part][sl],
                            ctx_x_eval, ctx_t_eval, ctx_y_eval)
            else:
                out = model(x[part][sl], t[part][sl])
            if task == "regression":
                preds.append(out.squeeze(-1).cpu().numpy())
            elif task == "binclass":
                preds.append(F.softmax(out, dim=-1)[:, 1].cpu().numpy())
            else:
                preds.append(F.softmax(out, dim=-1).cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        if task == "regression":
            pred = pred * y_std + y_mean
        return compute_metric(y_np[part], pred, task)

    higher_better = task != "regression"
    best = -np.inf if higher_better else np.inf
    best_epoch, since_improve, best_state = -1, 0, None
    val_history, epochs_run = [], 0

    pbar = tqdm(range(cfg.max_epochs), desc=f"{data.name}[{cfg.arch}/{cfg.time_mode}|{cfg.seed_tag}]",
                leave=False, dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        for idx in batches():
            xb, tb, yb = x["train"][idx], t["train"][idx], y["train"][idx]
            if needs_ctx:
                # in-batch retrieval: context = this batch, exclude the query itself.
                # need >= 2 instances for a non-self neighbor to exist.
                if xb.shape[0] < 2:
                    continue
                y_hat = model(xb, tb, xb, tb, yb,
                              exclude_self=torch.arange(xb.shape[0], device=device))
            else:
                y_hat = model(xb, tb)
            loss = main_loss(y_hat, yb)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        val = evaluate("val")
        epochs_run = epoch + 1
        if cfg.record_history:
            val_history.append(float(val))
        improved = (val > best) if higher_better else (val < best)
        pbar.set_postfix(val=f"{val:.4f}", best=f"{best if np.isfinite(best) else val:.4f}",
                         patience=f"{since_improve}/{cfg.patience}")
        if improved:
            best, best_epoch, since_improve = val, epoch, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            since_improve += 1
            # don't early-stop before min_epochs (zero-init time-modulation needs
            # epochs to train; stopping at epoch 0/1 makes time_tabr collapse to tabr).
            if epoch + 1 >= cfg.min_epochs and since_improve > cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_score = evaluate("test")
    return {
        "dataset": data.name, "task": task, "split": data.split,
        "arch": cfg.arch, "time_mode": cfg.time_mode, "time_basis": cfg.time_basis,
        "val_score": float(best), "score": float(test_score),
        "best_epoch": int(best_epoch), "n_epochs": int(epochs_run),
        "val_history": val_history if cfg.record_history else None, "model": model,
    }
