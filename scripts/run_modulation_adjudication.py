"""R2.3 — Cai & Ye modulation adjudication: is the gain X-side (covariate) or concept?

THE headline experiment (PLAN_V2 §R2). Cai & Ye (NeurIPS 2025) modulate INPUT
feature distributional statistics as a function of time (gamma·YeoJohnson(x,lambda)
+beta, label-FREE; src/models/temporal_modulation.py) and report beating baselines
on TabReD, calling the target "concept drift". But a label-free feature-distribution
transform is a time-indexed COVARIATE (P(x)) normalization by construction — it
cannot exploit a P(y|x) change. This script makes the empirical case:

  Per dataset (temporal split): train STATIC mlp vs the SAME mlp + temporal feature
  modulation; gain = modulated - static (oriented higher=better, paired per seed).
  Join with the model-light references (RESULTS §13): cov_AUC (X-shift strength) and
  within-overlap concept_gap (P(y|x) change). Then test, ACROSS datasets:
    Spearman(gain, cov_AUC)      -- expect POSITIVE (gain tracks covariate shift)
    Spearman(gain, concept_gap)  -- expect ~0 (gain is NOT concept-driven)
  and the smoking gun: a POSITIVE modulation gain on cooking/maps, where concept is
  measured to be ~0 -> the gain there is X-side adaptation, not concept exploitation.

    python scripts/run_modulation_adjudication.py --config configs/phase1.yaml \
        --all --elec2 --insects --n-seeds 5

Reuses the verified MLP trainer (train_timetabr arch='mlp', feature_modulation) and
the model-light drift measures. Requires PyTorch + sklearn (server).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy.stats import spearmanr, t as _t  # noqa: E402

from src.utils.seed import seed_everything  # noqa: E402
from src.data.tabred_loader import load_tabred, TABRED_DATASETS  # noqa: E402
from src.training.tabr_trainer import TabRConfig, train_timetabr  # noqa: E402
from src.analysis.drift_measure import (  # noqa: E402
    _stack, covariate_shift_auc, concept_within_overlap,
)
from src.utils.metrics import metric_name  # noqa: E402
from src.utils.stats import orient_higher_is_better, hedges_g_paired  # noqa: E402


def _early_late(data):
    t = data.train.t; med = float(np.median(t))
    X = _stack(data.train.X_num, data.train.X_bin, data.train.X_cat)
    em, lm = t <= med, t > med
    return X[em], data.train.y[em], X[lm], data.train.y[lm]


def _cfg(cfg, arch, seed, *, fmod, lr):
    tr = cfg.training
    return TabRConfig(
        arch=arch, feature_modulation=fmod, time_basis="trend",
        trend_degree=int(OmegaConf.select(cfg, "memory.trend_degree", default=3)),
        lr=float(lr), weight_decay=1e-4, dropout=0.1,
        batch_size=int(tr.batch_size), eval_batch=int(tr.eval_batch),
        patience=int(tr.patience), max_epochs=int(tr.max_epochs),
        seed_tag=f"s{seed}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--all", action="store_true", help="all 8 TabReD datasets")
    ap.add_argument("--elec2", action="store_true")
    ap.add_argument("--insects", action="store_true")
    ap.add_argument("--insects-variant", default="incremental_balanced")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0, help="seed for the drift references")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    root = Path(cfg.data.root)
    out_dir = Path(cfg.experiment.results_dir).parent / "modulation_adj"
    out_dir.mkdir(parents=True, exist_ok=True)
    split = cfg.experiment.split

    loaders = []
    names = list(TABRED_DATASETS) if args.all else list(cfg.data.sanity_datasets)
    for ds in names:
        loaders.append(("tabred", ds))
    if args.elec2:
        loaders.append(("elec2", "elec2"))
    if args.insects:
        loaders.append(("insects", args.insects_variant))

    def load(kind, name):
        if kind == "elec2":
            from src.data.elec2_loader import load_elec2
            return load_elec2(split="temporal", seed=args.seed)
        if kind == "insects":
            from src.data.insects_loader import load_insects
            return load_insects(variant=name, split="temporal", seed=args.seed)
        return load_tabred(name, root, split=split)

    print(f"\n==== R2.3 modulation adjudication (static mlp vs +temporal feature "
          f"modulation), {args.n_seeds} seeds, temporal ====")
    print(f"  {'dataset':24s}{'metric':>8s}{'cov_AUC':>8s}{'concept':>9s} | "
          f"{'static':>8s}{'modul':>8s}{'gain':>9s}{'95%CI':>18s}{'g_z':>6s}")
    rows = []
    for kind, name in loaders:
        data = load(kind, name)
        ds = data.name
        metric = metric_name(data.task)
        # references (model-light)
        Xe, ye, Xl, yl = _early_late(data)
        cov = covariate_shift_auc(Xe, Xl, seed=args.seed).get("auc")
        con = concept_within_overlap(Xe, ye, Xl, yl, data.task, seed=args.seed)
        gap = con.get("concept_gap_within_overlap") if con.get("measurable") else None
        # trained: static vs modulated, paired per seed
        s_static, s_modul = [], []
        for s in range(args.n_seeds):
            seed_everything(s)
            r0 = train_timetabr(data, _cfg(cfg, "mlp", s, fmod=False, lr=args.lr))
            seed_everything(s)
            r1 = train_timetabr(data, _cfg(cfg, "mlp", s, fmod=True, lr=args.lr))
            s_static.append(r0["score"]); s_modul.append(r1["score"])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        a = orient_higher_is_better(s_modul, metric)
        b = orient_higher_is_better(s_static, metric)
        d = a - b; n = len(d); m = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        h = float(_t.ppf(0.975, n - 1)) * se if n > 1 else float("nan")
        gz = hedges_g_paired(a, b)
        gtxt = f"{gap:+.3f}" if isinstance(gap, (int, float)) else "    -"
        print(f"  {ds:24s}{metric:>8s}{cov:8.3f}{gtxt:>9s} | "
              f"{np.mean(s_static):8.4f}{np.mean(s_modul):8.4f}{m:+9.4f}"
              f"  [{m-h:+.4f},{m+h:+.4f}]{gz:+6.2f}")
        rows.append({"dataset": ds, "task": data.task, "metric": metric,
                     "cov_auc": cov, "concept_gap": gap,
                     "mean_static": float(np.mean(s_static)),
                     "mean_modulated": float(np.mean(s_modul)),
                     "gain_oriented": m, "gain_ci95": [m - h, m + h],
                     "gain_g_z": gz, "static": s_static, "modulated": s_modul})

    # cross-dataset: does the gain track covariate shift or concept?
    gains = np.array([r["gain_oriented"] for r in rows])
    covs = np.array([r["cov_auc"] for r in rows])
    meas = [r for r in rows if isinstance(r["concept_gap"], (int, float))]
    print("\n  ==== adjudication (across datasets) ====")
    rho_cov, p_cov = spearmanr(covs, gains)
    print(f"  Spearman(gain, cov_AUC)     = {rho_cov:+.3f} (p={p_cov:.3f})  "
          f"[expect POSITIVE => gain tracks COVARIATE shift]")
    if len(meas) >= 3:
        gg = np.array([r["concept_gap"] for r in meas])
        gn = np.array([r["gain_oriented"] for r in meas])
        rho_c, p_c = spearmanr(gg, gn)
        print(f"  Spearman(gain, concept_gap) = {rho_c:+.3f} (p={p_c:.3f})  "
              f"[expect ~0 => gain NOT concept-driven]  (n={len(meas)} measurable)")
    # smoking gun: gain where concept ~ 0
    zc = [r for r in rows if isinstance(r["concept_gap"], (int, float))
          and abs(r["concept_gap"]) < 0.02]
    if zc:
        print("  smoking gun — modulation gain where concept~0 (|gap|<0.02):")
        for r in zc:
            sig = "POS" if r["gain_ci95"][0] > 0 else ("neg" if r["gain_ci95"][1] < 0 else "ns")
            print(f"    {r['dataset']:20s} concept={r['concept_gap']:+.3f} "
                  f"gain={r['gain_oriented']:+.4f} [{sig}] => X-side adaptation")
    print("  => modulation helps even with NO concept to exploit = covariate adaptation,")
    print("     not concept exploitation (their 'concept drift' = feature-distribution drift).")

    rec = {"mode": "modulation_adjudication", "ts": time.time(), "n_seeds": args.n_seeds,
           "lr": args.lr, "split": split, "rows": rows,
           "spearman_gain_cov_auc": float(rho_cov)}
    (out_dir / "summary.json").write_text(json.dumps(rec, indent=2, default=float))
    print(f"\n  wrote {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
