"""CPU smoke test for R2.3 temporal feature modulation (Cai&Ye reimpl) + 'mlp' arch.

Verifies:
1. yeo_johnson(x, lam=1) == x (identity power) and is finite for x<0, lam near 0/2.
2. TemporalModulation is the IDENTITY at init (gamma=1,beta=0,lambda=1) => fmod(x,t)≈x,
   and gradients reach gamma/beta/lambda heads after a step.
3. TimeTabRModel arch='mlp' (static) and arch='mlp'+feature_modulation: forward
   shapes ok, grad reaches encoder AND the modulation params.
4. train_timetabr arch='mlp' with feature_modulation False/True: finite scores.

    python scripts/smoke_test_modulation.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.models.temporal_modulation import TemporalModulation, yeo_johnson
from src.models.tabr import TimeTabRModel


def check_yeo_johnson():
    x = torch.linspace(-3, 3, 50)
    y1 = yeo_johnson(x, torch.ones_like(x))
    assert torch.allclose(y1, x, atol=1e-4), "YJ(x, lam=1) must be identity"
    for lam in (0.0, 1e-5, 2.0, 2 - 1e-5, -2.0, 4.0):
        y = yeo_johnson(x, torch.full_like(x, lam))
        assert torch.isfinite(y).all(), f"YJ non-finite at lam={lam}"
    print("yeo_johnson: identity at lam=1, finite across lam in [-2,4] OK")


def check_modulation_identity():
    torch.manual_seed(0)
    m = TemporalModulation(6)
    x = torch.randn(20, 6); t = torch.rand(20)
    out = m(x, t)
    assert torch.allclose(out, x, atol=1e-4), "modulation must be identity at init"
    out.pow(2).mean().backward()
    for nm, p in [("gamma", m.fc_gamma), ("beta", m.fc_beta), ("lambda", m.fc_lambda)]:
        assert p.weight.grad is not None and torch.isfinite(p.weight.grad).all(), nm
    print("TemporalModulation: identity at init + grads reach gamma/beta/lambda OK")


def check_model():
    B, nfeat = 16, 10
    x = torch.randn(B, nfeat); t = torch.rand(B); y = torch.randint(0, 2, (B,))
    for fmod in (False, True):
        m = TimeTabRModel(nfeat, "binclass", 2, arch="mlp", enc_dim=32,
                          feature_modulation=fmod)
        for p in m.parameters():
            p.grad = None
        out = m(x, t)
        assert out.shape == (B, 2), out.shape
        F.cross_entropy(out, y).backward()
        assert m.encoder[0].weight.grad is not None
        if fmod:
            assert m.fmod is not None and m.fmod.fc_lambda.weight.grad is not None, \
                "feature_modulation grad must flow"
        print(f"  arch=mlp feature_modulation={fmod}: out{tuple(out.shape)} grads OK")


def check_train():
    from src.data.tabred_loader import TabularSplit, TabReDDataset
    from src.training.tabr_trainer import TabRConfig, train_timetabr
    rng = np.random.default_rng(0); n, d = 800, 6
    X = rng.normal(size=(n, d)).astype("float32")
    t = np.sort(rng.uniform(0, 1, size=n)).astype("float32")
    # covariate shift over t (x mean drifts), FIXED rule => no concept (X-side only)
    X = X + (t[:, None] * 2.0)
    y = (X[:, 0] - t * 2.0 + 0.3 * rng.normal(size=n) > 0).astype("int64")

    def sp(lo, hi):
        sl = slice(lo, hi)
        return TabularSplit(X_num=X[sl], X_bin=None, X_cat=None, y=y[sl], t=t[sl],
                            t_raw=np.arange(lo, hi, dtype="int64"))
    a, b = int(0.7 * n), int(0.85 * n)
    data = TabReDDataset(name="synth", task="binclass", split="temporal",
                         train=sp(0, a), val=sp(a, b), test=sp(b, n), t_min=0.0, t_max=1.0)
    base = dict(enc_dim=32, enc_hidden=64, batch_size=64, eval_batch=128,
                max_epochs=5, patience=3, device="cpu")
    for fmod in (False, True):
        r = train_timetabr(data, TabRConfig(arch="mlp", feature_modulation=fmod,
                                            time_basis="trend", seed_tag="0", **base))
        assert np.isfinite(r["score"]), f"fmod={fmod} non-finite"
        print(f"  train arch=mlp feature_modulation={fmod}: "
              f"val={r['val_score']:.4f} test={r['score']:.4f} OK")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_yeo_johnson()
    check_modulation_identity()
    check_model()
    check_train()
    print("\nAll modulation (R2.3) smoke checks passed.")
