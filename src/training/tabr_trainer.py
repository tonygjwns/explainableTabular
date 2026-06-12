"""Training loop for the TimeTabRModel arms (Q2b structure axis, V2 protocol).

Trains TimeTabRModel (shared MLP encoder; arch in {mlp_t, tabr, tabr_t, time_tabr,
time_tabr_t}) on one dataset/seed with a single supervised loss (CE / MSE). This is
the runner-side half of the Q2b infra; the model + retrieval core live in
src/models/tabr.py.

Retrieval protocol — V2 (external audit 2026-06-12; see PLAN_V2.md):
- TRAIN (train_context='sampled', default): context = [the minibatch] + a fresh
  per-step sample of `train_context_size` OTHER train instances (batch indices
  excluded so a query can never meet a duplicate of itself), exclude_self over the
  batch positions. This closes the pre-V2 train/eval mismatch (in-batch 255
  candidates at train vs 4096 at eval) AND keeps the Δt input distribution of the
  hooks comparable between train and eval.
  train_context='inbatch' reproduces the legacy protocol (context = batch only).
- EVAL (eval_context='full', default): the candidate pool is the ENTIRE train set
  (computationally trivial at these scales; pre-V2 used an arbitrary 13% uniform
  subsample). eval_context='fixed' reproduces the legacy fixed sample of
  `eval_context_size`, now drawn from a DEDICATED generator seeded by `ctx_seed`
  so all arms at the same seed share the same pool (pre-V2 drew from the global
  RNG after model construction → pools differed across arms).
  The context is re-encoded once per evaluate() call (weights change), no_grad.

arch='mlp_t' ignores the context entirely (time is just a feature) — so the same
loop trains all arms; the factorial runner only flips cfg.arch / cfg.time_mode.
Batches with <2 samples are skipped for ALL arms (pre-V2 skipped them only for
retrieval arms — an arm-asymmetric training stream).

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
    arch: str = "time_tabr_t"        # mlp_t | tabr | tabr_t | time_tabr | time_tabr_t
    time_mode: str = "value"         # value | metric | both (time_tabr* arms only)
    value_hook: str = "mlp"          # mlp | gate | linear(LEGACY — collapses, see tabr.py)
    time_basis: str = "trend"        # trend (extrapolation-safe) | fourier
    trend_degree: int = 3
    time_out: int = 16
    # shared encoder
    enc_dim: int = 128
    enc_hidden: int = 256
    n_enc_layers: int = 2
    dropout: float = 0.0         # encoder dropout (regularize so models train past
                                 # the early-overfit peak; applied to ALL arms)
    # retrieval (V2 substrate)
    topk: int = 32
    sim_scale: str = "sqrt_d"        # sqrt_d (learnable τ) | none (LEGACY raw -‖·‖²)
    key_proj: bool = True            # learned key projection (False = LEGACY raw z)
    predictor_hidden: int = 256
    train_context: str = "sampled"   # sampled (batch + fresh sample) | inbatch (LEGACY)
    train_context_size: int = 4096   # extra candidates per step (train_context='sampled')
    eval_context: str = "full"       # full train pool | fixed (LEGACY subsample)
    eval_context_size: int = 4096    # pool size when eval_context='fixed'
    ctx_seed: int = 0                # dedicated RNG seed for the fixed eval pool
                                     # (shared across arms at the same run seed)
    # optimization
    lr: float = 2e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    eval_batch: int = 1024
    patience: int = 16            # in EVAL events (per-epoch, or per eval_every_steps)
    min_epochs: int = 0          # floor before early-stop may fire (give zero-init
                                 # time-modulation room to train; 0 = off)
    eval_every_steps: int = 0    # 0 = eval once per epoch; >0 = eval every N optimizer
                                 # steps (resolution: catch a peak inside epoch 1)
    max_epochs: int = 200
    device: str = "cuda"
    seed_tag: str = ""
    record_history: bool = False  # collect per-epoch val into the result (diagnostic)


def _build_features(data: TabReDDataset):
    """Prepared numeric (quantile X_num + X_bin) with categoricals appended one-hot.

    The encoder takes a single flat float matrix, so categoricals are one-hot
    encoded with GLOBAL per-column cardinalities (a temporal test split can hold
    codes unseen in train; the global max sizes the one-hot WIDTH only — purely
    structural, no value statistics leak). Returns ({part: float32 ndarray}, n_features).
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
        # train∪val only (no test peek). Accuracy on a test class unseen in train
        # is still well-defined (the model just can't predict it).
        n_classes = int(max(int(y_np["train"].max()), int(y_np["val"].max())) + 1)
        y = {p: torch.as_tensor(y_np[p], dtype=torch.long, device=device) for p in y_np}

    model = TimeTabRModel(
        n_features, task, n_classes, arch=cfg.arch, time_mode=cfg.time_mode,
        enc_dim=cfg.enc_dim, enc_hidden=cfg.enc_hidden, n_enc_layers=cfg.n_enc_layers,
        time_basis=cfg.time_basis, trend_degree=cfg.trend_degree, time_out=cfg.time_out,
        topk=cfg.topk, predictor_hidden=cfg.predictor_hidden, dropout=cfg.dropout,
        value_hook=cfg.value_hook, sim_scale=cfg.sim_scale, key_proj=cfg.key_proj,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_train = x["train"].shape[0]
    needs_ctx = cfg.arch != "mlp_t"

    # eval context pool: full train (V2 default) or a fixed legacy subsample drawn
    # from a DEDICATED generator (same pool for every arm at the same run seed).
    if needs_ctx:
        if cfg.eval_context == "full" or n_train <= cfg.eval_context_size:
            ctx_x_eval, ctx_t_eval, ctx_y_eval = x["train"], t["train"], y["train"]
        else:
            gen = torch.Generator().manual_seed(int(cfg.ctx_seed))
            ctx_idx = torch.randperm(n_train, generator=gen)[: cfg.eval_context_size].to(device)
            ctx_x_eval = x["train"][ctx_idx]
            ctx_t_eval, ctx_y_eval = t["train"][ctx_idx], y["train"][ctx_idx]

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
        # encode the (possibly full-train) context ONCE per evaluate() call
        zc = model.encode(ctx_x_eval) if needs_ctx else None
        for i in range(0, n, cfg.eval_batch):
            sl = slice(i, i + cfg.eval_batch)
            if needs_ctx:
                out = model.tabr(model.encode(x[part][sl]), t[part][sl],
                                 zc, ctx_t_eval, ctx_y_eval)
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
    val_history, train_loss_history, epochs_run = [], [], 0

    pbar = tqdm(range(cfg.max_epochs), desc=f"{data.name}[{cfg.arch}/{cfg.time_mode}|{cfg.seed_tag}]",
                leave=False, dynamic_ncols=True)

    def record_eval(epoch) -> bool:
        """Eval val, update best / early-stop state. Returns True if training should stop.
        patience counts EVAL events; min_epochs floors the stop in epoch units."""
        nonlocal best, best_epoch, since_improve, best_state
        val = evaluate("val")
        model.train()                       # evaluate() flipped to eval(); resume
        if cfg.record_history:
            val_history.append(float(val))
        improved = (val > best) if higher_better else (val < best)
        if improved:
            best, best_epoch, since_improve = val, epoch, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            since_improve += 1
        pbar.set_postfix(val=f"{val:.4f}", best=f"{best if np.isfinite(best) else val:.4f}",
                         patience=f"{since_improve}/{cfg.patience}")
        return epoch + 1 >= cfg.min_epochs and since_improve > cfg.patience

    stop = False
    for epoch in pbar:
        model.train()
        ep_loss, nb = 0.0, 0
        for idx in batches():
            xb, tb, yb = x["train"][idx], t["train"][idx], y["train"][idx]
            # need >= 2 instances for a non-self neighbor; skip for ALL arms so the
            # training stream is identical across arms (arm-symmetric).
            if xb.shape[0] < 2:
                continue
            if needs_ctx:
                if cfg.train_context == "sampled":
                    # context = batch + fresh sample of OTHER train instances
                    # (batch excluded from the extra pool: a duplicate of the query
                    # would evade exclude_self = self-label leakage).
                    mask = torch.ones(n_train, dtype=torch.bool, device=device)
                    mask[idx] = False
                    pool = mask.nonzero(as_tuple=True)[0]
                    m = min(cfg.train_context_size, pool.shape[0])
                    extra = pool[torch.randperm(pool.shape[0], device=device)[:m]]
                    cx = torch.cat([xb, x["train"][extra]], dim=0)
                    ct = torch.cat([tb, t["train"][extra]], dim=0)
                    cy = torch.cat([yb, y["train"][extra]], dim=0)
                else:                       # 'inbatch' (LEGACY): context = the batch
                    cx, ct, cy = xb, tb, yb
                y_hat = model(xb, tb, cx, ct, cy,
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
            ep_loss += float(loss); nb += 1
            if cfg.eval_every_steps and nb % cfg.eval_every_steps == 0:
                if record_eval(epoch):
                    stop = True
                    break
        epochs_run = epoch + 1
        if cfg.record_history:
            train_loss_history.append(ep_loss / max(nb, 1))
        if not cfg.eval_every_steps and record_eval(epoch):
            stop = True
        if stop:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_score = evaluate("test")
    return {
        "dataset": data.name, "task": task, "split": data.split,
        "arch": cfg.arch, "time_mode": cfg.time_mode, "time_basis": cfg.time_basis,
        "value_hook": cfg.value_hook,
        "val_score": float(best), "score": float(test_score),
        "best_epoch": int(best_epoch), "n_epochs": int(epochs_run),
        "val_history": val_history if cfg.record_history else None,
        "train_loss_history": train_loss_history if cfg.record_history else None,
        "model": model,
    }
