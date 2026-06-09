"""Q1 GATE — functional faithfulness on trend-representable binary synthetic.

PLAN_RESCUE §A (final protocol). Tests the MAIN claim falsifiably:
- synthetic: y = 1[x·w(t)+noise > 0]  (BINARY, matches V_k=label-embedding; logit
  linear in x). w(t) = cos(α)w1 + sin(α)w2 with α = (π/2)·t  → MONOTONE 0→π/2
  (≤ half period) so it is TREND-representable (no basis-mismatch confound).
- model under test: Phase1Model, memory_only + trend basis + load-balance, inject off
  (time only via the memory).
- metric: recovery = mean_t cos(w_hat(t), w_true(t)), w_hat(t)=E_x[∂(logit₁−logit₀)/∂x].
- ceiling hi = MLP-on-[x,t] (realizable ceiling; should be ~1, else gradient/synth bug).
  floor lo = same model trained with SHUFFLED t (t–y link broken).
- gate (band on the ACROSS-SEED distribution): PASS if ≥8/10 seeds ≥ PASS-line
  (= lo+0.7(hi−lo)); FAIL→(다) if mean < FAIL-line (= lo+0.4(hi−lo)); else band (cheap
  follow-up). Also plots recovery(t) (t→1 collapse previews Q2 extrapolation trouble).

    python scripts/run_q1_faithfulness.py
    python scripts/run_q1_faithfulness.py --lb 0.0 0.003 0.01 0.03   # load-balance sweep
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.utils.seed import seed_everything  # noqa: E402
from src.data.tabred_loader import TabularSplit, TabReDDataset  # noqa: E402
from src.training.phase1_trainer import Phase1Config, train_phase1  # noqa: E402
from src.analysis.faithfulness import recovery_curve, band_line  # noqa: E402


# ---------- synthetic (trend-representable, binary) ----------
def make_synth(n, d, sigma, seed, shuffle_t=False):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype("float32")
    t = rng.random(n).astype("float32")
    w1 = rng.standard_normal(d); w1 /= np.linalg.norm(w1)
    w2 = rng.standard_normal(d); w2 -= (w2 @ w1) * w1; w2 /= np.linalg.norm(w2)  # ⊥ w1
    alpha = (np.pi / 2.0) * t                                   # monotone 0→π/2
    wt = np.cos(alpha)[:, None] * w1[None] + np.sin(alpha)[:, None] * w2[None]
    logit = (x * wt).sum(1) + sigma * rng.standard_normal(n)
    y = (logit > 0).astype("int64")
    t_used = (rng.permutation(t) if shuffle_t else t).astype("float32")  # shuffle = floor
    return x, y, t_used, w1, w2


def w_true_fn(w1, w2):
    return lambda tv: np.cos(np.pi / 2 * tv) * w1 + np.sin(np.pi / 2 * tv) * w2


def to_ds(x, y, t, seed):
    n = len(y); idx = np.random.default_rng(seed + 1).permutation(n)
    ntr, nva = int(0.7 * n), int(0.15 * n)
    mk = lambda ii: TabularSplit(X_num=x[ii], X_bin=None, X_cat=None, y=y[ii],
                                 t=t[ii], t_raw=(t[ii] * 1e6).astype("int64"))
    return TabReDDataset(name="q1synth", task="binclass", split="random",
                         train=mk(idx[:ntr]), val=mk(idx[ntr:ntr + nva]),
                         test=mk(idx[ntr + nva:]), t_min=0.0, t_max=1.0)


def pcfg(lb):
    return Phase1Config(
        k=8, n_blocks=2, d_block=128, dropout=0.0,
        n_prototypes=200, rank=16, mem_hidden=64, tau_temp=0.3, predictor_hidden=128,
        predictor_mode="memory_only", time_indexed=True, inject_time_input=False,
        mem_time_out_dim=16, n_harmonics=4, time_periods=(1.0,),
        time_basis="trend", trend_degree=3, load_balance_coef=lb,
        kmeans_init=True, n_slices=5, kmeans_max_samples=8000, lambda_smooth=0.0,
        lr=2e-3, batch_size=256, eval_batch=4096, patience=12, max_epochs=80, seed_tag="q1")


def phase1_score(model):
    def f(xb, tb):
        out = model(xb, None, tb)          # (M, 2)
        return out[:, 1] - out[:, 0]
    return f


# ---------- ceiling: MLP on [x, t] ----------
class MLPt(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, h), nn.ReLU(), nn.Linear(h, h),
                                 nn.ReLU(), nn.Linear(h, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, t.reshape(-1, 1)], dim=1))


def train_mlpt(x, y, t, device, epochs=60):
    m = MLPt(x.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3)
    X = torch.tensor(x, device=device); T = torch.tensor(t, device=device)
    Y = torch.tensor(y, device=device)
    for _ in range(epochs):
        perm = torch.randperm(len(Y), device=device)
        for i in range(0, len(Y), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            F.cross_entropy(m(X[idx], T[idx]), Y[idx]).backward()
            opt.step()
    m.eval()
    return m


def mlpt_score(m):
    def f(xb, tb):
        out = m(xb, tb)
        return out[:, 1] - out[:, 0]
    return f


def model_recovery(score_fn, w1, w2, t_grid, x_eval, device):
    rec, _ = recovery_curve(score_fn, w_true_fn(w1, w2), t_grid, x_eval, device)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--T", type=int, default=50)
    ap.add_argument("--M", type=int, default=512)
    ap.add_argument("--lb", type=float, nargs="+", default=[0.01],
                    help="load_balance coef(s); multiple -> sweep")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_grid = np.linspace(0.0, 1.0, args.T)
    x_eval = np.random.default_rng(0).standard_normal((args.M, args.d)).astype("float32")
    out_dir = Path("results/phase1/q1"); out_dir.mkdir(parents=True, exist_ok=True)

    # ceiling (MLP+t) and floor (shuffle-t model), averaged over 3 seeds
    hi_list, lo_list = [], []
    for s in range(3):
        xc, yc, tc, w1, w2 = make_synth(args.n, args.d, args.sigma, 100 + s)
        seed_everything(100 + s)
        mlp = train_mlpt(xc, yc, tc, device)
        hi_list.append(model_recovery(mlpt_score(mlp), w1, w2, t_grid, x_eval, device).mean())
        xf, yf, tf, w1f, w2f = make_synth(args.n, args.d, args.sigma, 200 + s, shuffle_t=True)
        seed_everything(200 + s)
        mf = train_phase1(to_ds(xf, yf, tf, 200 + s), pcfg(args.lb[0]))["model"]
        lo_list.append(model_recovery(phase1_score(mf), w1f, w2f, t_grid, x_eval, device).mean())
    hi, lo = float(np.mean(hi_list)), float(np.mean(lo_list))
    pass_line, fail_line = band_line(lo, hi, 0.7), band_line(lo, hi, 0.4)
    print(f"ceiling(MLP+t)={hi:.3f}  floor(shuffle-t)={lo:.3f}  "
          f"PASS-line={pass_line:.3f}  FAIL-line={fail_line:.3f}")
    if hi < 0.9:
        print("  ⚠ ceiling < 0.9 → gradient/synth sanity issue (expected ~1).")

    results = {"hi": hi, "lo": lo, "pass_line": pass_line, "fail_line": fail_line,
               "sigma": args.sigma, "sweep": {}}
    rec_t_repr = None
    for lb in args.lb:
        recs = []
        for s in range(args.seeds):
            x, y, t, w1, w2 = make_synth(args.n, args.d, args.sigma, s)
            seed_everything(s)
            model = train_phase1(to_ds(x, y, t, s), pcfg(lb))["model"]
            rec_curve, _ = recovery_curve(phase1_score(model), w_true_fn(w1, w2),
                                          t_grid, x_eval, device)
            recs.append(float(rec_curve.mean()))
            if lb == args.lb[0] and s == 0:
                rec_t_repr = rec_curve.tolist()
            print(f"[lb={lb} seed{s}] recovery(mean over t)={recs[-1]:.3f}")
        recs = np.array(recs)
        n_pass = int((recs >= pass_line).sum())
        lower = float(recs.mean() - 1.96 * recs.std(ddof=1) / np.sqrt(len(recs)))
        verdict = ("PASS" if n_pass >= max(1, int(0.8 * args.seeds))
                   else ("FAIL" if recs.mean() < fail_line else "BAND(inconclusive)"))
        results["sweep"][str(lb)] = {"recoveries": recs.tolist(), "mean": float(recs.mean()),
                                     "lower95": lower, "n_pass": n_pass, "verdict": verdict}
        print(f"==== lb={lb}: mean={recs.mean():.3f} lower95={lower:.3f} "
              f"n_pass={n_pass}/{args.seeds} → {verdict} ====")

    results["recovery_t_curve_repr"] = rec_t_repr
    (out_dir / "q1_verdict.json").write_text(json.dumps(results, indent=2))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(t_grid, rec_t_repr, "-o", ms=3)
        plt.axhline(pass_line, color="g", ls="--", label="PASS line")
        plt.axhline(fail_line, color="r", ls="--", label="FAIL line")
        plt.xlabel("t (normalized)"); plt.ylabel("recovery cos(ŵ(t), w(t))")
        plt.title("Q1 functional faithfulness recovery(t)"); plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / "recovery_t.png", dpi=120)
        print(f"saved plot -> {out_dir/'recovery_t.png'}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    print(f"saved -> {out_dir/'q1_verdict.json'}")


if __name__ == "__main__":
    main()
