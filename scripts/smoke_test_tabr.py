"""CPU smoke test for the TimeTabR retrieval infra (V2: hooks, arms, expressivity).

Verifies wiring (shapes, top-k, time_mode toggles, gradient flow into the enabled
hook) AND two V2 guarantees (PLAN_V2.md meta-rule: hypothesis→math→code equivalence):

1. check_linear_collapse — the EXPRESSIVITY regression test. With uniform retrieval
   weights, permuting per-neighbor times (same multiset) must change the output for
   value_hook ∈ {mlp, gate} (per-neighbor label×time interaction survives
   aggregation) and must NOT change it for the legacy 'linear' hook
   (Σ w·Linear(Δτ) = Linear(Σ w·Δτ) — the collapse that invalidated the pre-V2
   negative; audit 2026-06-12).
2. check_init_equivalence — zero-init hooks: time_mode='both' must equal
   time_mode='none' at init under the same seed (clean within-arch time ablation),
   for every hook and for concat_time on/off.

    python scripts/smoke_test_tabr.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.models.tabr import TimeTabR, TimeTabRModel


def _hook_last_weight(model: TimeTabR):
    if model.value_hook == "linear":
        return model.value_time.weight
    if model.value_hook == "mlp":
        return model.value_time_mlp[-1].weight
    return model.value_gate[-1].weight


def check(task, n_classes=3, value_hook="mlp"):
    B, N, d = 16, 200, 24
    print(f"\n=== task={task} hook={value_hook} ===")
    zq = torch.randn(B, d); tq = torch.rand(B)
    zc = torch.randn(N, d); tc = torch.rand(N)
    yc = (torch.rand(N) if task == "regression"
          else torch.randint(0, (n_classes if task == "multiclass" else 2), (N,)))
    yq = (torch.randn(B) if task == "regression"
          else torch.randint(0, (n_classes if task == "multiclass" else 2), (B,)))
    for tm in ("none", "metric", "value", "both"):
        model = TimeTabR(d, task, n_classes, topk=8, time_mode=tm, value_hook=value_hook)
        for p in model.parameters():
            p.grad = None
        out, aux = model(zq, tq, zc, tc, yc, return_aux=True)
        exp = n_classes if task == "multiclass" else (2 if task == "binclass" else 1)
        assert out.shape == (B, exp), f"{out.shape} != {(B, exp)}"
        assert aux["w"].shape == (B, 8)
        assert torch.allclose(aux["w"].sum(1), torch.ones(B), atol=1e-5)
        loss = (F.mse_loss(out.squeeze(-1), yq.float()) if task == "regression"
                else F.cross_entropy(out, yq))
        loss.backward()
        g_val = model.value[0].weight.grad is not None
        g_lab = (model.label_emb.weight.grad is not None
                 if hasattr(model.label_emb, "weight") else True)
        hw = _hook_last_weight(model)
        g_vt = hw.grad is not None and float(hw.grad.abs().sum()) > 0
        g_mt = (model.metric_time.weight.grad is not None
                and float(model.metric_time.weight.grad.abs().sum()) > 0)
        g_tau = (model.log_tau.grad is not None and float(model.log_tau.grad.abs()) > 0)
        time_grad = {"none": "(n/a)", "metric": g_mt, "value": g_vt,
                     "both": (g_vt and g_mt)}[tm]
        print(f"  time_mode={tm:6s}: out{tuple(out.shape)} w_sum=1 OK  "
              f"grad[value={g_val},label={g_lab},tau={g_tau}] time_mod_grad={time_grad}")
        if tm != "none":
            assert time_grad is True, f"{tm}/{value_hook}: no grad into the enabled hook"


def check_linear_collapse():
    """EXPRESSIVITY GUARD: linear hook collapses under aggregation; mlp/gate must not.

    Construction: all candidates at the same point => uniform retrieval weights w.
    Permuting per-neighbor times keeps Σ w·Δτ fixed, so the linear hook's output is
    invariant (the collapse identity), while mlp/gate change (label×time pairing
    matters). Hook weights are un-zeroed so the test is non-trivial.
    """
    B, k, d = 2, 4, 16
    torch.manual_seed(0)
    zq = torch.randn(B, d); tq = torch.rand(B)
    zc = torch.randn(1, d).repeat(k, 1)              # identical candidates -> uniform w
    yc = torch.tensor([0, 1, 0, 1])
    tc1 = torch.tensor([0.10, 0.20, 0.30, 0.40])
    tc2 = torch.tensor([0.40, 0.30, 0.20, 0.10])     # permuted: same multiset/mean
    print("\n=== linear-collapse expressivity guard ===")
    for hook, must_collapse in (("linear", True), ("mlp", False), ("gate", False)):
        torch.manual_seed(1)
        m = TimeTabR(d, "binclass", 2, topk=k, time_mode="value", value_hook=hook)
        with torch.no_grad():                         # un-zero the hook
            w = _hook_last_weight(m)
            w.normal_(0, 0.5)
        m.eval()
        with torch.no_grad():
            o1 = m(zq, tq, zc, tc1, yc)
            o2 = m(zq, tq, zc, tc2, yc)
        same = torch.allclose(o1, o2, atol=1e-5)
        verdict = "COLLAPSES (aggregated-Δt feature)" if same else "per-neighbor (survives aggregation)"
        print(f"  hook={hook:6s}: permuted-times output {'un' if same else ''}changed -> {verdict}")
        assert same == must_collapse, (
            f"value_hook='{hook}': expected collapse={must_collapse}, got {same} — "
            "the hook no longer matches its documented expressivity (see tabr.py header)")


def check_init_equivalence():
    """Zero-init hooks: time_mode='both' ≡ 'none' at init (same seed), all hooks."""
    B, N, d = 8, 64, 16
    torch.manual_seed(2)
    zq = torch.randn(B, d); tq = torch.rand(B)
    zc = torch.randn(N, d); tc = torch.rand(N)
    yc = torch.randint(0, 2, (N,))
    print("\n=== init equivalence (zero-init hooks) ===")
    for hook in ("linear", "mlp", "gate"):
        for concat in (False, True):
            ms = []
            for tm in ("none", "both"):
                torch.manual_seed(7)
                ms.append(TimeTabR(d, "binclass", 2, topk=8, time_mode=tm,
                                   value_hook=hook, concat_time=concat))
            with torch.no_grad():
                o = [m(zq, tq, zc, tc, yc) for m in ms]
            assert torch.allclose(o[0], o[1], atol=1e-6), \
                f"hook={hook} concat_time={concat}: time_mode both != none at init"
            print(f"  hook={hook:6s} concat_time={concat}: none == both at init OK")


def check_exclude_self():
    # in-batch retrieval: candidates == queries, must exclude self
    B, d = 12, 16
    z = torch.randn(B, d); t = torch.rand(B); y = torch.randint(0, 2, (B,))
    m = TimeTabR(d, "binclass", 2, topk=4, time_mode="both")
    out = m(z, t, z, t, y, exclude_self=torch.arange(B))
    assert out.shape == (B, 2)
    # V2 sampled-context style: context = [batch ; extra], self at positions 0..B-1
    extra_z = torch.randn(40, d); extra_t = torch.rand(40); extra_y = torch.randint(0, 2, (40,))
    out = m(z, t, torch.cat([z, extra_z]), torch.cat([t, extra_t]),
            torch.cat([y, extra_y]), exclude_self=torch.arange(B))
    assert out.shape == (B, 2)
    print("\nexclude_self OK (in-batch + sampled-context layouts)")


def check_model(task, n_classes=3):
    B, Nc, nfeat = 16, 300, 10
    xq = torch.randn(B, nfeat); tq = torch.rand(B)
    xc = torch.randn(Nc, nfeat); tc = torch.rand(Nc)
    yc = (torch.rand(Nc) if task == "regression"
          else torch.randint(0, (n_classes if task == "multiclass" else 2), (Nc,)))
    yq = (torch.randn(B) if task == "regression"
          else torch.randint(0, (n_classes if task == "multiclass" else 2), (B,)))
    exp = n_classes if task == "multiclass" else (2 if task == "binclass" else 1)
    print(f"\n=== model task={task} ===")
    for arch, tm in [("mlp_t", "none"), ("tabr", "none"), ("tabr_t", "none"),
                     ("time_tabr", "value"), ("time_tabr_t", "value"),
                     ("time_tabr_t", "both")]:
        m = TimeTabRModel(nfeat, task, n_classes, arch=arch, time_mode=tm, enc_dim=32, topk=8)
        for p in m.parameters():
            p.grad = None
        out = m(xq, tq, xc, tc, yc)
        assert out.shape == (B, exp), f"{arch}/{tm}: {out.shape} != {(B, exp)}"
        loss = (F.mse_loss(out.squeeze(-1), yq.float()) if task == "regression"
                else F.cross_entropy(out, yq))
        loss.backward()
        g_enc = m.encoder[0].weight.grad is not None
        print(f"  arch={arch:11s} time_mode={tm:5s}: out{tuple(out.shape)} OK, grad[enc={g_enc}]")
    # in-batch retrieval (candidates == queries)
    m = TimeTabRModel(nfeat, task, n_classes, arch="time_tabr_t", time_mode="value",
                      enc_dim=32, topk=4)
    _ = m(xq, tq, xq, tq, yq, exclude_self=torch.arange(B))
    print("  in-batch (exclude_self) OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    for task in ("binclass", "multiclass", "regression"):
        check(task)                       # default hook (mlp)
    check("binclass", value_hook="gate")
    check("binclass", value_hook="linear")
    check_linear_collapse()
    check_init_equivalence()
    check_exclude_self()
    for task in ("binclass", "multiclass", "regression"):
        check_model(task)
    print("\nAll TabR smoke checks passed (V2: hooks/arms/expressivity/init-equivalence).")
