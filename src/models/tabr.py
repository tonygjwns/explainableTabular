"""Instance-based retrieval (TabR-style) with TIME + LABEL hooks — F2-independent infra.

The core of Q2b. Unlike the prototype memory (learned V_k = no new info), the value
carries the RETRIEVED NEIGHBORS' REAL LABELS. Critically, the aggregation exposes,
per neighbor, ALL of (t_q, t_i, y_i) so EITHER F2 choice drops in without a rebuild:
  - metric-side : modulate similarity by trend(t_q)  (which neighbors)         [covariate adapt]
  - value-side  : correct the neighbor's label contribution by (t_i→t_q) drift  [concept exploit]
`time_mode ∈ {none(=TabR), metric, value, both}`. The time-modulation terms are
ZERO-INITIALIZED → at init the model ≈ plain TabR; training grows whichever is enabled.
This lets the SAME architecture toggle time on/off (2-stage ablation) with zero
architecture confound, and lets F2 be finalized (post-alignment) by setting time_mode
+ refining the modulation parametrization — no replumbing.

Efficiency: top-k retrieval, so per-neighbor value (B,k,d) with (t_q,t_i) deps is cheap.
Operates on ENCODINGS z (caller encodes query + a candidate/context set with a shared
encoder, and supplies candidate labels/times). Requires PyTorch.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_embedding import FourierTimeEmbedding


class TimeTabR(nn.Module):
    def __init__(
        self, dim: int, task: str, n_classes: int = 2, *,
        time_basis: str = "trend", trend_degree: int = 3, time_out: int = 16,
        topk: int = 32, value_hidden: int = 128, predictor_hidden: int = 256,
        time_mode: str = "none",
    ):
        super().__init__()
        self.task = task.lower()
        self.dim = dim
        self.topk = topk
        self.time_mode = time_mode
        out_dim = n_classes if self.task == "multiclass" else (2 if self.task == "binclass" else 1)

        # value carries the neighbor's real LABEL
        if self.task == "regression":
            self.label_emb = nn.Linear(1, dim)
        else:
            self.n_classes = n_classes
            self.label_emb = nn.Embedding(n_classes, dim)

        self.time_emb = FourierTimeEmbedding(time_out, basis=time_basis,
                                             trend_degree=trend_degree, use_trend=True)
        td = self.time_emb.out_dim
        # value: MLP over [label_emb(y_i) ; z_q - z_i]
        self.value = nn.Sequential(nn.Linear(2 * dim, value_hidden), nn.ReLU(),
                                   nn.Linear(value_hidden, dim))
        # value-side (t_i->t_q) label-drift correction (ZERO-INIT -> starts as TabR)
        self.value_time = nn.Linear(td, dim)
        nn.init.zeros_(self.value_time.weight); nn.init.zeros_(self.value_time.bias)
        # metric-side time modulation gated by trend(t_q) (ZERO-INIT)
        self.metric_time = nn.Linear(td, dim)
        nn.init.zeros_(self.metric_time.weight); nn.init.zeros_(self.metric_time.bias)
        self.predictor = nn.Sequential(nn.Linear(2 * dim, predictor_hidden), nn.ReLU(),
                                       nn.Linear(predictor_hidden, out_dim))

    def _label_vec(self, y: torch.Tensor) -> torch.Tensor:
        if self.task == "regression":
            return self.label_emb(y.float().reshape(-1, 1))
        return self.label_emb(y.long())

    def forward(self, zq, tq, zc, tc, yc, *, exclude_self: Optional[torch.Tensor] = None,
                return_aux: bool = False):
        """zq (B,d), tq (B,); candidates zc (N,d), tc (N,), yc (N,)."""
        B, d = zq.shape
        N = zc.shape[0]
        sq = (zq * zq).sum(1, keepdim=True)                 # (B,1)
        sc = (zc * zc).sum(1).unsqueeze(0)                  # (1,N)
        sim = -(sq - 2.0 * zq @ zc.t() + sc)                # (B,N) = -||zq-zc||^2
        if exclude_self is not None:
            sim = sim.scatter(1, exclude_self.view(-1, 1), float("-inf"))
        k = min(self.topk, N - (1 if exclude_self is not None else 0))
        topv, topi = sim.topk(k, dim=1)                     # (B,k)
        z_nbr = zc[topi]                                    # (B,k,d)
        y_nbr = yc[topi]                                    # (B,k)
        t_nbr = tc[topi]                                    # (B,k)

        if self.time_mode in ("metric", "both"):            # HOOK: t_q in similarity
            gq = self.metric_time(self.time_emb(tq))        # (B,d)
            topv = topv + (gq.unsqueeze(1) * z_nbr).sum(-1)  # (B,k)
        w = F.softmax(topv, dim=1)                          # (B,k)

        lab = self._label_vec(y_nbr.reshape(-1)).reshape(B, k, d)        # HOOK: y_i
        val = self.value(torch.cat([lab, zq.unsqueeze(1).expand(B, k, d) - z_nbr], dim=-1))
        if self.time_mode in ("value", "both"):             # HOOK: (t_i -> t_q) drift
            dtime = (self.time_emb(tq).unsqueeze(1).expand(B, k, -1)
                     - self.time_emb(t_nbr.reshape(-1)).reshape(B, k, -1))
            val = val + self.value_time(dtime)              # zero-init -> starts as TabR
        agg = (w.unsqueeze(-1) * val).sum(1)                # (B,d)
        out = self.predictor(torch.cat([zq, agg], dim=-1))
        if return_aux:
            return out, {"w": w, "topi": topi}
        return out


class TimeTabRModel(nn.Module):
    """3-arm factorial model (shared encoder): the structure axis of Q2b.

      arch='mlp_t'    : predictor([z ; trend(t)])            — time as a FEATURE (baseline)
      arch='tabr'     : TimeTabR retrieval, time_mode='none' — non-parametric, no time
      arch='time_tabr': TimeTabR retrieval, time_mode=value/metric/both — the candidate

    Same encoder + same time basis across arms (so 'structure vs feature' and 'basis'
    are cleanly separable). Caller supplies a CONTEXT set (features, t, y) of training
    instances for retrieval (in-batch during training, a fixed sample at eval).
    Encoder is a plain MLP over the prepared numeric matrix (num+bin); categoricals
    can be appended one-hot by the caller. Requires PyTorch.
    """

    def __init__(self, n_features: int, task: str, n_classes: int = 2, *,
                 arch: str = "time_tabr", time_mode: str = "value",
                 enc_dim: int = 128, enc_hidden: int = 256, n_enc_layers: int = 2,
                 time_basis: str = "trend", trend_degree: int = 3, time_out: int = 16,
                 topk: int = 32, predictor_hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.task = task.lower(); self.arch = arch
        out_dim = n_classes if self.task == "multiclass" else (2 if self.task == "binclass" else 1)
        # shared encoder; dropout (>0) regularizes ALL arms identically so they can
        # train past the early-overfit peak (lets time_tabr's drift correction engage).
        layers, prev = [], n_features
        for _ in range(n_enc_layers):
            layers += [nn.Linear(prev, enc_hidden), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = enc_hidden
        layers += [nn.Linear(prev, enc_dim)]
        self.encoder = nn.Sequential(*layers)
        if arch == "mlp_t":
            self.time_emb = FourierTimeEmbedding(time_out, basis=time_basis,
                                                 trend_degree=trend_degree, use_trend=True)
            self.predictor = nn.Sequential(
                nn.Linear(enc_dim + self.time_emb.out_dim, predictor_hidden), nn.ReLU(),
                nn.Linear(predictor_hidden, out_dim))
        else:
            tm = "none" if arch == "tabr" else time_mode
            self.tabr = TimeTabR(enc_dim, task, n_classes, time_basis=time_basis,
                                 trend_degree=trend_degree, time_out=time_out, topk=topk,
                                 predictor_hidden=predictor_hidden, time_mode=tm)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, xq, tq, ctx_x=None, ctx_t=None, ctx_y=None,
                exclude_self=None, return_aux=False):
        zq = self.encode(xq)
        if self.arch == "mlp_t":
            out = self.predictor(torch.cat([zq, self.time_emb(tq)], dim=-1))
            return (out, {}) if return_aux else out
        zc = self.encode(ctx_x)
        return self.tabr(zq, tq, zc, ctx_t, ctx_y,
                         exclude_self=exclude_self, return_aux=return_aux)
