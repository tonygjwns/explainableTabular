"""DECIDER: does the time-indexed memory beat fixed on REAL concept drift (Elec2)?

Synthetic proved the mechanism works; TabReD proved its drift is covariate-only.
Elec2 is the canonical real concept-drift tabular benchmark — the bridge. If
time-indexed > fixed here, the positive (A') path is real.

Per seed: train time_indexed vs fixed (paired), compare with Wilcoxon + Hedges' g.
inject_time_input=False (config) so the ONLY time pathway is the memory; with
split='random' all t are covered in train (cleanest "exploits concept drift" test).

    python scripts/run_elec2.py --config configs/phase1.yaml --split random
    python scripts/run_elec2.py --config configs/phase1.yaml --split temporal --predictor-mode residual
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.utils.seed import seed_everything  # noqa: E402
from src.data.elec2_loader import load_elec2  # noqa: E402
from src.training.phase1_trainer import train_phase1  # noqa: E402
from src.utils.stats import orient_higher_is_better, paired_wilcoxon, hedges_g  # noqa: E402
from run_phase1_sanity import make_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--split", default="random", choices=["random", "temporal"])
    ap.add_argument("--predictor-mode", default=None,
                    choices=["concat", "memory_only", "residual"])
    ap.add_argument("--inject", action="store_true",
                    help="inject time at input too (with time_indexed=False this is the "
                         "time-as-FEATURE baseline -> tests if the memory STRUCTURE is needed)")
    ap.add_argument("--diag-every", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.predictor_mode:
        cfg.memory.predictor_mode = args.predictor_mode
    if args.inject:
        cfg.memory.inject_time_input = True
    seeds = list(cfg.experiment.seeds)
    out_dir = Path(cfg.experiment.results_dir).parent / "elec2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Elec2 decider: split={args.split}, predictor_mode={cfg.memory.predictor_mode}, "
          f"inject={cfg.memory.inject_time_input}, seeds={seeds}")

    st, sf = [], []
    for seed in seeds:
        seed_everything(seed)
        data = load_elec2(split=args.split, seed=seed)
        rt = train_phase1(data, _cfg(cfg, seed, True, args.diag_every))
        rf = train_phase1(data, _cfg(cfg, seed, False, args.diag_every))
        a, b = float(rt["score"]), float(rf["score"])
        st.append(a); sf.append(b)
        print(f"[elec2 s{seed}] roc_auc: time={a:.4f}  fixed={b:.4f}  diff={a - b:+.4f}")
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    a = orient_higher_is_better(st, "roc_auc")
    b = orient_higher_is_better(sf, "roc_auc")
    p = paired_wilcoxon(a, b)
    g = hedges_g(a, b)
    delta = float(np.mean(a) - np.mean(b))
    verdict = {"split": args.split, "predictor_mode": cfg.memory.predictor_mode,
               "seeds": seeds, "time": st, "fixed": sf,
               "mean_time": float(np.mean(st)), "mean_fixed": float(np.mean(sf)),
               "delta_oriented": delta, "p_value": p, "hedges_g": g,
               "positive": bool(delta > 0 and p < 0.05 and g >= 0.3)}
    (out_dir / f"elec2_{args.split}_{cfg.memory.predictor_mode}.json").write_text(json.dumps(verdict, indent=2))

    print("\n==== Elec2 decider ====")
    print(f"  mean AUC: time={np.mean(st):.4f}  fixed={np.mean(sf):.4f}  "
          f"delta={delta:+.4f}  p={p:.4f}  g={g:+.2f}")
    print(f"  POSITIVE (time>fixed, p<0.05, g>=0.3): {verdict['positive']}")
    print("  -> True  : mechanism exploits REAL concept drift => A' path is live")
    print("  -> False : even on a canonical concept-drift benchmark it doesn't help")


def _cfg(cfg, seed, time_indexed, diag):
    c = make_cfg(cfg, seed, time_indexed)
    c.diag_every = diag
    return c


if __name__ == "__main__":
    main()
