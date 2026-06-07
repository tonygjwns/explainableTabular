"""Synthetic POSITIVE CONTROL: does the time-indexed memory work when it MUST?

Validates implementation correctness before trusting any TabReD null. We build a
dataset with PURE concept drift and NO covariate drift:

    x ~ N(0, I)                      (stationary -> no P(x) shift)
    t ~ U[0, 1]
    w(t) = cos(2πt) w1 + sin(2πt) w2 (a rotating weight vector)
    y = x · w(t) + noise             (P(y|x,t) rotates with t -> pure concept drift)

A time-blind model can only learn E_t[w(t)] = 0 -> predicts ~0 -> RMSE ~ std(y).
A model that uses t (here ONLY via the time-indexed memory P_k(t)) can recover
w(t) -> low RMSE. Split is RANDOM (all t covered in train), so the memory can
learn the mapping across the whole range.

EXPECTED if the implementation is correct:
    time_indexed=True  RMSE  <<  time_indexed=False (~ std(y))
    and mem_gap (diag) clearly > 0.
If time-indexed CANNOT beat fixed here, the time/memory pathway is broken (bug),
and the TabReD null is uninformative.

    python scripts/run_synth_control.py                       # default concat, inject off
    python scripts/run_synth_control.py --predictor-mode memory_only
"""
from __future__ import annotations

import argparse
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.seed import seed_everything  # noqa: E402
from src.data.tabred_loader import TabularSplit, TabReDDataset  # noqa: E402
from src.training.phase1_trainer import Phase1Config, train_phase1  # noqa: E402


def make_synth(n, d, noise, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype("float32")
    t = rng.random(n).astype("float32")
    w1 = rng.standard_normal(d); w1 /= np.linalg.norm(w1)
    w2 = rng.standard_normal(d); w2 /= np.linalg.norm(w2)
    wt = (np.cos(2 * np.pi * t)[:, None] * w1[None, :]
          + np.sin(2 * np.pi * t)[:, None] * w2[None, :])      # (n, d)
    y = ((x * wt).sum(1) + noise * rng.standard_normal(n)).astype("float32")
    return x, y, t


def split_ds(x, y, t, seed):
    n = len(t)
    idx = np.random.default_rng(seed + 1).permutation(n)
    n_tr, n_va = int(0.7 * n), int(0.15 * n)

    def mk(ii):
        return TabularSplit(X_num=x[ii], X_bin=None, X_cat=None, y=y[ii],
                            t=t[ii], t_raw=(t[ii] * 1e6).astype("int64"))
    return TabReDDataset(name="synth_conceptdrift", task="regression", split="random",
                         train=mk(idx[:n_tr]), val=mk(idx[n_tr:n_tr + n_va]),
                         test=mk(idx[n_tr + n_va:]), t_min=0.0, t_max=1.0)


def cfg(time_indexed, predictor_mode, inject, diag):
    return Phase1Config(
        k=8, n_blocks=2, d_block=128, dropout=0.0,
        n_prototypes=200, rank=16, mem_hidden=64, tau_temp=0.3,
        predictor_hidden=128, predictor_mode=predictor_mode,
        time_indexed=time_indexed, inject_time_input=inject,
        input_time_out_dim=8, mem_time_out_dim=16, n_harmonics=4, time_periods=(1.0,),
        kmeans_init=True, n_slices=5, kmeans_max_samples=8000, lambda_smooth=0.01,
        lr=2e-3, batch_size=256, eval_batch=4096, patience=12, max_epochs=80,
        diag_every=diag, seed_tag="synth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--predictor-mode", default="concat",
                    choices=["concat", "memory_only", "residual"])
    ap.add_argument("--inject", action="store_true",
                    help="also inject time at input (default: memory is the ONLY time path)")
    ap.add_argument("--diag-every", type=int, default=10)
    args = ap.parse_args()

    seed_everything(args.seed)
    x, y, t = make_synth(args.n, args.d, args.noise, args.seed)
    data = split_ds(x, y, t, args.seed)
    std_y = float(np.std(data.test.y))
    print(f"synth concept-drift: n={args.n} d={args.d} noise={args.noise}  "
          f"std(y_test)={std_y:.4f}  (time-blind RMSE should be ~this)\n")

    out = {}
    for ti in (True, False):
        print(f"--- time_indexed={ti}  predictor_mode={args.predictor_mode}  inject={args.inject} ---")
        seed_everything(args.seed)
        res = train_phase1(data, cfg(ti, args.predictor_mode, args.inject, args.diag_every))
        out["time" if ti else "fixed"] = res["score"]
        print(f"    => RMSE={res['score']:.4f}\n")

    time_rmse, fixed_rmse = out["time"], out["fixed"]
    print("==== Synthetic positive control ====")
    print(f"  std(y)         = {std_y:.4f}   (trivial time-blind baseline)")
    print(f"  fixed  RMSE    = {fixed_rmse:.4f}")
    print(f"  time   RMSE    = {time_rmse:.4f}")
    print(f"  improvement    = {(fixed_rmse - time_rmse) / fixed_rmse:+.1%} (time vs fixed)")
    ok = time_rmse < 0.6 * fixed_rmse
    print(f"  IMPLEMENTATION {'OK (time-indexed exploits concept drift)' if ok else 'SUSPECT (time cannot beat fixed -> bug in time/memory path)'}")


if __name__ == "__main__":
    main()
