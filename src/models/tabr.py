"""Instance-based retrieval (TabR-style) with TIME + LABEL hooks — Q2b core (V2).

The value carries the RETRIEVED NEIGHBORS' REAL LABELS; the aggregation exposes,
per neighbor, ALL of (t_q, t_i, y_i) so either F2 choice drops in:
  - metric-side : modulate similarity by trend(t_q)  (which neighbors)         [covariate adapt]
  - value-side  : correct the neighbor's label contribution by (t_i→t_q) drift  [concept exploit]
`time_mode ∈ {none(=TabR), metric, value, both}`. Time-modulation terms are
ZERO-INITIALIZED → at init the model ≈ plain TabR, so time on/off is a clean
within-architecture ablation.

V2 changes (external audit 2026-06-12 — see PLAN_V2.md):
1. value_hook ∈ {linear, mlp, gate}. The legacy 'linear' hook COLLAPSES under
   aggregation:  Σ_k w_k·Linear(Δτ_k) = Linear(Σ_k w_k·Δτ_k)  — independent of y_i,
   i.e. it reduces to ONE aggregated Δt feature and CANNOT express a per-neighbor
   stale-label correction (the hypothesis Q2b claims to test). 'mlp' (label×time
   interaction, additive) and 'gate' (multiplicative discount of the value) survive
   aggregation. 'linear' is kept only to reproduce pre-V2 runs.
2. Similarity scaling: sim_scale='sqrt_d' divides -‖·‖² by √d·τ with a LEARNABLE
   temperature τ (legacy 'none' = raw squared L2 → softmax sharpness at the mercy
   of encoder scale; the prototype side tuned τ but Q2b never had one).
3. key_proj: a learned key projection decouples the retrieval metric from the
   predictor's z (TabR-faithful). Legacy: similarity on raw z.
4. metric-side modulation is applied BEFORE top-k (it can now change WHICH
   neighbors are retrieved; legacy applied it after selection → could only
   reweight the time-agnostic top-k).
5. concat_time: optionally concatenate τ(t_q) into the predictor input. This
   builds the missing arms tabr_t / time_tabr_t: retrieval arms get the SAME
   direct time feature as mlp_t, so "structure vs feature" is no longer
   confounded with "where time enters".

Expressivity guard (smoke_test_tabr.py::check_linear_collapse): with uniform
retrieval weights, permuting per-neighbor times must change the output for
mlp/gate hooks and must NOT for the linear hook.

Efficiency: top-k retrieval, so per-neighbor value (B,k,d) with (t_q,t_i) deps is
cheap. Operates on ENCODINGS z (caller encodes query + a candidate/context set with
a shared encoder, and supplies candidate labels/times). Requires PyTorch.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .temporal_embedding import FourierTimeEmbedding

VALUE_HOOKS = ("linear", "mlp", "gate")


class TimeTabR(nn.Module):
    def __init__(
        self, dim: int, task: str, n_classes: int = 2, *,
        time_basis: str = "trend", trend_degree: int = 3, time_out: int = 16,
        topk: int = 32, value_hidden: int = 128, predictor_hidden: int = 256,
        time_mode: str = "none", value_hook: str = "mlp",
        sim_scale: str = "sqrt_d", key_proj: bool = True, concat_time: bool = False,
    ):
        super().__init__()
        if value_hook not in VALUE_HOOKS:
            raise ValueError(f"value_hook must be one of {VALUE_HOOKS}, got {value_hook!r}")
        if sim_scale not in ("none", "sqrt_d"):
            raise ValueError(f"sim_scale must be 'none' or 'sqrt_d', got {sim_scale!r}")
        self.task = task.lower()
        self.dim = dim
        self.topk = topk
        self.time_mode = time_mode
        self.value_hook = value_hook
        self.sim_scale = sim_scale
        self.concat_time = concat_time
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
        # retrieval metric: learned key projection (decoupled from predictor z) + temperature
        self.key = nn.Linear(dim, dim) if key_proj else None
        if sim_scale == "sqrt_d":
            self.log_tau = nn.Parameter(torch.zeros(()))   # τ=1 at init; sim = -‖·‖²/(√d·τ)
        # value: MLP over [label_emb(y_i) ; z_q - z_i]
        self.value = nn.Sequential(nn.Linear(2 * dim, value_hidden), nn.ReLU(),
                                   nn.Linear(value_hidden, dim))
        # ---- value-side (t_i->t_q) label-drift correction (ZERO-INIT -> starts as TabR) ----
        # NOTE: modules are created based on value_hook (NOT time_mode), so at a fixed
        # value_hook the parameter creation order — hence init under a fixed seed — is
        # identical across time_mode settings (clean within-architecture time ablation).
        if value_hook == "linear":      # LEGACY — collapses to an aggregated Δt feature
            self.value_time = nn.Linear(td, dim)
            nn.init.zeros_(self.value_time.weight); nn.init.zeros_(self.value_time.bias)
        elif value_hook == "mlp":       # per-neighbor label×time interaction (additive)
            self.value_time_mlp = nn.Sequential(
                nn.Linear(dim + td, value_hidden), nn.ReLU(), nn.Linear(value_hidden, dim))
            nn.init.zeros_(self.value_time_mlp[-1].weight)
            nn.init.zeros_(self.value_time_mlp[-1].bias)
        else:                            # 'gate': multiplicative stale-label discount
            self.value_gate = nn.Sequential(
                nn.Linear(td, value_hidden), nn.ReLU(), nn.Linear(value_hidden, dim))
            nn.init.zeros_(self.value_gate[-1].weight)
            nn.init.zeros_(self.value_gate[-1].bias)     # gate = 1+0 at init
        # metric-side time modulation gated by trend(t_q) (ZERO-INIT)
        self.metric_time = nn.Linear(td, dim)
        nn.init.zeros_(self.metric_time.weight); nn.init.zeros_(self.metric_time.bias)
        pred_in = 2 * dim + (td if concat_time else 0)
        self.predictor = nn.Sequential(nn.Linear(pred_in, predictor_hidden), nn.ReLU(),
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
        kq = self.key(zq) if self.key is not None else zq
        kc = self.key(zc) if self.key is not None else zc
        sq = (kq * kq).sum(1, keepdim=True)                 # (B,1)
        sc = (kc * kc).sum(1).unsqueeze(0)                  # (1,N)
        sim = -(sq - 2.0 * kq @ kc.t() + sc)                # (B,N) = -||kq-kc||^2
        if self.sim_scale == "sqrt_d":
            sim = sim / (math.sqrt(d) * self.log_tau.exp())
        if self.time_mode in ("metric", "both"):            # HOOK: t_q in similarity,
            gq = self.metric_time(self.time_emb(tq))        # BEFORE top-k → can change
            sim = sim + gq @ kc.t()                         # WHICH neighbors are retrieved
        if exclude_self is not None:
            sim = sim.scatter(1, exclude_self.view(-1, 1), float("-inf"))
        k = min(self.topk, N - (1 if exclude_self is not None else 0))
        topv, topi = sim.topk(k, dim=1)                     # (B,k)
        z_nbr = zc[topi]                                    # (B,k,d)
        y_nbr = yc[topi]                                    # (B,k)
        t_nbr = tc[topi]                                    # (B,k)
        w = F.softmax(topv, dim=1)                          # (B,k)

        lab = self._label_vec(y_nbr.reshape(-1)).reshape(B, k, d)        # HOOK: y_i
        val = self.value(torch.cat([lab, zq.unsqueeze(1).expand(B, k, d) - z_nbr], dim=-1))
        if self.time_mode in ("value", "both"):             # HOOK: (t_i -> t_q) drift
            dtime = (self.time_emb(tq).unsqueeze(1).expand(B, k, -1)
                     - self.time_emb(t_nbr.reshape(-1)).reshape(B, k, -1))
            if self.value_hook == "linear":                 # legacy (collapses; see header)
                val = val + self.value_time(dtime)
            elif self.value_hook == "mlp":                  # label×Δt interaction
                val = val + self.value_time_mlp(torch.cat([lab, dtime], dim=-1))
            else:                                           # 'gate': discount stale values
                val = val * (1.0 + self.value_gate(dtime))
        agg = (w.unsqueeze(-1) * val).sum(1)                # (B,d)
        pred_in = [zq, agg]
        if self.concat_time:                                # tabr_t / time_tabr_t arms
            pred_in.append(self.time_emb(tq))
        out = self.predictor(torch.cat(pred_in, dim=-1))
        if return_aux:
            return out, {"w": w, "topi": topi}
        return out


class TimeTabRModel(nn.Module):
    """Factorial model (shared encoder): the structure axis of Q2b (V2 = 5 arms).

      arch='mlp_t'      : predictor([z ; τ(t)])              — time as a FEATURE (baseline)
      arch='tabr'       : retrieval, no time anywhere        — isolates the substrate
      arch='tabr_t'     : retrieval + τ(t) at the predictor  — substrate ON TOP of the feature
      arch='time_tabr'  : retrieval + time hooks (legacy candidate; time only via hooks)
      arch='time_tabr_t': retrieval + time hooks + τ(t) at the predictor — the V2 candidate

    PRIMARY contrast (PREREG_V2): time_tabr_t − tabr_t = the pure contribution of
    time-INDEXING the retrieval, with the direct time feature held present in both
    arms (pre-V2 conflated 'structure vs feature' with 'where time enters').

    Same encoder + same time basis across arms. Caller supplies a CONTEXT set
    (features, t, y) of training instances for retrieval. Encoder is a plain MLP
    over the prepared numeric matrix (num+bin; categoricals appended one-hot by
    the caller). Requires PyTorch.
    """

    ARCHS = ("mlp_t", "tabr", "tabr_t", "time_tabr", "time_tabr_t")

    def __init__(self, n_features: int, task: str, n_classes: int = 2, *,
                 arch: str = "time_tabr_t", time_mode: str = "value",
                 enc_dim: int = 128, enc_hidden: int = 256, n_enc_layers: int = 2,
                 time_basis: str = "trend", trend_degree: int = 3, time_out: int = 16,
                 topk: int = 32, predictor_hidden: int = 256, dropout: float = 0.0,
                 value_hook: str = "mlp", sim_scale: str = "sqrt_d", key_proj: bool = True):
        super().__init__()
        if arch not in self.ARCHS:
            raise ValueError(f"arch must be one of {self.ARCHS}, got {arch!r}")
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
            tm = "none" if arch in ("tabr", "tabr_t") else time_mode
            self.tabr = TimeTabR(enc_dim, task, n_classes, time_basis=time_basis,
                                 trend_degree=trend_degree, time_out=time_out, topk=topk,
                                 predictor_hidden=predictor_hidden, time_mode=tm,
                                 value_hook=value_hook, sim_scale=sim_scale,
                                 key_proj=key_proj,
                                 concat_time=arch in ("tabr_t", "time_tabr_t"))

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
