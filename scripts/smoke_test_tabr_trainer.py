"""CPU smoke test for train_timetabr (the Q2b runner-side loop).

Builds a tiny synthetic TabReDDataset (with a planted concept: P(y|x) flips over t)
and runs all three arms {mlp_t, tabr, time_tabr} for a few epochs on CPU. Verifies
the full pipeline wires up — feature prep (numeric + one-hot cat), in-batch retrieval
training, fixed-context eval — and returns finite scores. Not a performance test.

    python scripts/smoke_test_tabr_trainer.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.tabred_loader import TabularSplit, TabReDDataset
from src.training.tabr_trainer import TabRConfig, train_timetabr


def _synth(n=900, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype("float32")
    cat = rng.integers(0, 4, size=(n, 1)).astype("int64")
    t = np.sort(rng.uniform(0, 1, size=n)).astype("float32")
    # planted concept: the sign of the decision boundary flips across time
    w = X[:, 0] + X[:, 1]
    logit = np.where(t < 0.5, w, -w) + 0.3 * rng.normal(size=n)
    y = (logit > 0).astype("int64")
    return X, cat, t, y


def _split(X, cat, t, y, lo, hi):
    sl = slice(lo, hi)
    return TabularSplit(X_num=X[sl], X_bin=None, X_cat=cat[sl], y=y[sl],
                        t=t[sl], t_raw=np.arange(lo, hi, dtype="int64"))


def main():
    X, cat, t, y = _synth()
    n = len(y); a, b = int(0.7 * n), int(0.85 * n)
    data = TabReDDataset(
        name="synth", task="binclass", split="temporal",
        train=_split(X, cat, t, y, 0, a),
        val=_split(X, cat, t, y, a, b),
        test=_split(X, cat, t, y, b, n),
        t_min=0.0, t_max=1.0,
    )

    base = dict(enc_dim=32, enc_hidden=64, topk=8, eval_context_size=256,
                batch_size=64, eval_batch=128, max_epochs=4, patience=3, device="cpu")
    for arch, tm in [("mlp_t", "none"), ("tabr", "none"),
                     ("time_tabr", "value"), ("time_tabr", "both")]:
        r = train_timetabr(data, TabRConfig(arch=arch, time_mode=tm, time_basis="trend",
                                            seed_tag="0", **base))
        assert np.isfinite(r["score"]), f"{arch}/{tm}: non-finite score"
        assert np.isfinite(r["val_score"]), f"{arch}/{tm}: non-finite val"
        print(f"  arch={arch:9s} time_mode={tm:5s}: "
              f"val={r['val_score']:.4f} test={r['score']:.4f} best_epoch={r['best_epoch']} OK")

    print("\ntrain_timetabr smoke checks passed (all arms finite).")


if __name__ == "__main__":
    main()
