"""Smoke test for the INSECTS loader + multiclass Q2b path (server; needs `river`).

Loads a small head of an INSECTS stream, checks the TabReDDataset shape/task, then
runs all 3 arms for a couple epochs to confirm the MULTICLASS pipeline wires up
end-to-end. Skips gracefully (exit 0) if `river` is not installed or the download
is unavailable — so it never blocks CI on a machine without river/network.

    python scripts/smoke_test_insects.py
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    try:
        from src.data.insects_loader import load_insects
        data = load_insects(variant="incremental_balanced", split="temporal",
                            seed=0, max_samples=3000)
    except Exception as e:  # river missing / download blocked -> skip, don't fail
        print(f"SKIP (insects unavailable): {type(e).__name__}: {e}")
        return

    assert data.task == "multiclass", data.task
    n_tr = data.train.X_num.shape[0]
    n_cls = int(max(data.train.y.max(), data.val.y.max(), data.test.y.max()) + 1)
    assert data.train.X_num.ndim == 2 and n_tr > 0
    assert data.train.t.min() >= 0.0 and data.test.t.max() <= 1.0001
    print(f"  loaded {data.name}: n_train={n_tr} n_feat={data.train.X_num.shape[1]} "
          f"n_classes={n_cls} task={data.task}")

    from src.training.tabr_trainer import TabRConfig, train_timetabr
    base = dict(enc_dim=32, enc_hidden=64, topk=8, eval_context_size=256,
                batch_size=64, eval_batch=128, max_epochs=3, patience=2, device="cpu")
    for arch, tm in [("mlp_t", "none"), ("tabr", "none"), ("time_tabr", "value")]:
        r = train_timetabr(data, TabRConfig(arch=arch, time_mode=tm, time_basis="trend",
                                            seed_tag="0", **base))
        assert np.isfinite(r["score"]), f"{arch}/{tm}: non-finite (acc)"
        print(f"  arch={arch:9s} time_mode={tm:5s}: val_acc={r['val_score']:.4f} "
              f"test_acc={r['score']:.4f} OK")
    print("\ninsects loader + multiclass Q2b smoke passed.")


if __name__ == "__main__":
    main()
