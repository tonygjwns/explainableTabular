"""Phase-1 training loop for the assembled time-indexed memory model.

Trains Phase1Model end-to-end with a single loss (EXPERIMENT_PLAN §5, §6):

    L = L_main(y_hat, y) + lambda_smooth * L_smooth

- L_main: cross-entropy (classification) / MSE on standardized target (regression)
- L_smooth: prototype trajectory smoothness (0 when time_indexed=False)

Differences from the Phase-0 trainer (src/training/trainer.py):
- model is Phase1Model (TabM encode -> memory retrieval), NOT raw TabM, so there is
  NO k-submodel mean-loss; the loss is the ordinary loss on y_hat.
- timestamps t are batched alongside x (memory is indexed by t).
- retrieval materializes a (B, K, d) tensor, so BOTH training and evaluation run in
  mini-batches (a full-split forward would OOM at K=1000).
- prototypes are KMeans-initialized in z-space before training (decision 5).

Reuses the Phase-0 numeric prep / cat-cardinality helpers for consistency.
Requires PyTorch + sklearn + the tabm package (runs on the server).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..data.tabred_loader import TabReDDataset
from ..models.phase1_model import Phase1Model
from ..models.proto_init import time_sliced_kmeans_init
from ..models.retrieval import load_balance_loss
from ..utils.metrics import compute_metric
from .trainer import _prep_numeric, _to_tensor, _global_cat_cardinalities
from .diagnostics import grad_norms, forward_diagnostics, format_line


@dataclass
class Phase1Config:
    # backbone (TabM)
    k: int = 32
    n_blocks: int = 3
    d_block: int = 512
    dropout: float = 0.1
    arch_type: str = "tabm"
    # memory / retrieval
    n_prototypes: int = 1000
    rank: int = 32
    mem_hidden: int = 64
    tau_temp: float = 1.0
    predictor_hidden: int = 256
    predictor_mode: str = "concat"       # concat | memory_only | residual (force memory use)
    time_indexed: bool = True            # False -> fixed-memory control (Test 1)
    inject_time_input: bool = True       # decision 4 auxiliary path (Test 4 toggle)
    input_time_out_dim: int = 8
    mem_time_out_dim: int = 32
    n_harmonics: int = 4
    # Fourier periods in the SAME units as the normalized t (loader maps the
    # training range to [0,1]). (1.0,) => fundamental = the whole span; with
    # n_harmonics that is a smooth low-frequency basis (1..n cycles over [0,1]).
    # The old (1, 1/12, 1/52, 1/365) assumed "t==years" and is ultra-high-freq
    # on [0,1] -> oscillatory drift. (calendar-aware time is a later refinement.)
    time_periods: tuple = (1.0,)
    time_basis: str = "fourier"          # 'fourier' | 'trend' (extrapolation-safe poly)
    trend_degree: int = 3
    load_balance_coef: float = 0.0       # >0: anti-collapse (Switch-style, keeps per-query sharp)
    # KMeans init (decision 5)
    kmeans_init: bool = True
    n_slices: int = 10
    kmeans_max_samples: int = 50_000
    # loss
    lambda_smooth: float = 1.0
    # optimization
    lr: float = 2e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    eval_batch: int = 1024
    patience: int = 16
    max_epochs: int = 200
    device: str = "cuda"
    seed_tag: str = ""
    diag_every: int = 0          # 0=off; else print internal diagnostics every N epochs
    diag_sample: int = 2048      # sample size for the diagnostics probes


def train_phase1(data: TabReDDataset, cfg: Phase1Config) -> dict:
    """Train Phase1Model on one dataset/seed.

    Returns a result dict {score, val_score, best_epoch, ...} and the trained
    model under key 'model' (for retrieval / trajectory analysis in sanity tests).
    """
    device = cfg.device if torch.cuda.is_available() else "cpu"
    task = data.task

    # ---- features (quantile-normalize X_num + concat X_bin; impute NaN) ----
    (xnum_tr, xnum_va, xnum_te), n_num = _prep_numeric(data.train, data.val, data.test)
    cat_card = _global_cat_cardinalities(data.train.X_cat, data.val.X_cat, data.test.X_cat)

    x_num = {p: _to_tensor(a, torch.float32, device)
             for p, a in zip(("train", "val", "test"), (xnum_tr, xnum_va, xnum_te))}
    x_cat = {p: _to_tensor(getattr(data, p).X_cat, torch.long, device)
             for p in ("train", "val", "test")}
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
        n_classes = int(max(int(y_np["train"].max()), int(y_np["val"].max())) + 1)
        y = {p: torch.as_tensor(y_np[p], dtype=torch.long, device=device) for p in y_np}

    # ---- model ----
    model = Phase1Model(
        n_num_features=n_num, cat_cardinalities=cat_card, task=task, n_classes=n_classes,
        k=cfg.k, n_blocks=cfg.n_blocks, d_block=cfg.d_block, dropout=cfg.dropout,
        arch_type=cfg.arch_type,
        n_prototypes=cfg.n_prototypes, rank=cfg.rank, mem_hidden=cfg.mem_hidden,
        tau_temp=cfg.tau_temp, predictor_hidden=cfg.predictor_hidden,
        predictor_mode=cfg.predictor_mode,
        time_indexed=cfg.time_indexed, inject_time_input=cfg.inject_time_input,
        input_time_out_dim=cfg.input_time_out_dim, mem_time_out_dim=cfg.mem_time_out_dim,
        n_harmonics=cfg.n_harmonics, time_periods=tuple(cfg.time_periods),
        time_basis=cfg.time_basis, trend_degree=cfg.trend_degree,
    ).to(device)

    # ---- KMeans prototype init in z-space (decision 5) ----
    if cfg.kmeans_init:
        base = time_sliced_kmeans_init(
            model, xnum_tr, data.train.X_cat, data.train.t,
            K=cfg.n_prototypes, n_slices=cfg.n_slices,
            max_samples=cfg.kmeans_max_samples, device=device, seed=0,
        )
        model.init_memory_from_kmeans(base)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_train = (x_num["train"] if x_num["train"] is not None else x_cat["train"]).shape[0]

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
        n = (x_num[part] if x_num[part] is not None else x_cat[part]).shape[0]
        preds = []
        for i in range(0, n, cfg.eval_batch):
            sl = slice(i, i + cfg.eval_batch)
            xb_num = None if x_num[part] is None else x_num[part][sl]
            xb_cat = None if x_cat[part] is None else x_cat[part][sl]
            out = model(xb_num, xb_cat, t[part][sl])         # (b, out_dim)
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

    pbar = tqdm(range(cfg.max_epochs), desc=f"{data.name}[{cfg.seed_tag}]",
                leave=False, dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        for idx in batches():
            xb_num = None if x_num["train"] is None else x_num["train"][idx]
            xb_cat = None if x_cat["train"] is None else x_cat["train"][idx]
            tb = t["train"][idx]
            if cfg.load_balance_coef > 0:
                y_hat, aux = model(xb_num, xb_cat, tb, return_aux=True)
            else:
                y_hat, aux = model(xb_num, xb_cat, tb), None
            loss = main_loss(y_hat, y["train"][idx])
            if cfg.lambda_smooth > 0:
                loss = loss + cfg.lambda_smooth * model.smoothness_penalty(tb)
            if aux is not None:
                loss = loss + cfg.load_balance_coef * load_balance_loss(aux["w"])
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        if cfg.diag_every and (epoch % cfg.diag_every == 0):
            di = torch.randperm(n_train, device=device)[:cfg.diag_sample]
            dn = None if x_num["train"] is None else x_num["train"][di]
            dc = None if x_cat["train"] is None else x_cat["train"][di]
            model.train()
            dloss = main_loss(model(dn, dc, t["train"][di]), y["train"][di])
            if cfg.lambda_smooth > 0:
                dloss = dloss + cfg.lambda_smooth * model.smoothness_penalty(t["train"][di])
            opt.zero_grad(); dloss.backward()
            gn = grad_norms(model)
            opt.zero_grad(set_to_none=True)
            nv = (x_num["val"] if x_num["val"] is not None else x_cat["val"]).shape[0]
            vi = torch.arange(min(cfg.diag_sample, nv), device=device)
            vn = None if x_num["val"] is None else x_num["val"][vi]
            vc = None if x_cat["val"] is None else x_cat["val"][vi]
            fs = forward_diagnostics(model, vn, vc, t["val"][vi], y["val"][vi], task)
            tqdm.write(format_line(epoch, fs, gn))

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
        "time_indexed": cfg.time_indexed, "inject_time_input": cfg.inject_time_input,
        "val_score": float(best), "score": float(test_score),
        "best_epoch": int(best_epoch), "model": model,
    }
