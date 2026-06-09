"""Q2b factorial: does STRUCTURE (time-TabR) beat time-as-a-FEATURE on measured concept?

Elec2 is the one dataset with a measured, exploitable concept (+0.132 within-overlap).
Q2 asks whether the time-TabR STRUCTURE exploits it better than a basis-matched
time-feature MLP. Factorial design (NEXT_TAB.md ★):

    {arch: mlp_t, tabr, time_tabr} x {time_basis: fourier, trend} x {split: temporal, random} x seed

Three arms share the SAME encoder + time basis (clean 'structure vs feature' / 'basis'):
  - mlp_t     : predictor([z ; basis(t)])         — time as a FEATURE (the baseline to beat)
  - tabr      : retrieval, time_mode=none         — structure, NO time (isolates 'time')
  - time_tabr : retrieval, time_mode=value        — structure + value-side label-drift hook

Pre-registered reads (PLAN_RESCUE / NEXT_TAB):
  - A structure gain must show on the TEMPORAL split (early->late stale labels) and be
    ~0 on RANDOM (train sees both periods). Gain only on random => red flag (not concept).
  - Decisive contrasts (paired Wilcoxon + Hedges' g over seeds):
      time_tabr(value) vs mlp_t  (basis-matched) : does structure carry time better?
      time_tabr        vs tabr                   : does adding the time hook help at all?

    python scripts/run_elec2_q2.py --config configs/phase1.yaml --n-seeds 25
    python scripts/run_elec2_q2.py --config configs/phase1.yaml --n-seeds 25 \
        --splits temporal --bases trend --time-mode value
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
from src.training.tabr_trainer import TabRConfig, train_timetabr  # noqa: E402
from src.utils.metrics import metric_name  # noqa: E402
from src.utils.stats import (  # noqa: E402
    orient_higher_is_better, paired_wilcoxon, hedges_g,
)


def _cfg(cfg, arch, time_mode, time_basis, seed) -> TabRConfig:
    tr = cfg.training
    topk = OmegaConf.select(cfg, "tabr.topk", default=32)
    return TabRConfig(
        arch=arch, time_mode=time_mode, time_basis=time_basis,
        trend_degree=int(OmegaConf.select(cfg, "memory.trend_degree", default=3)),
        topk=int(topk),
        lr=float(tr.learning_rate), weight_decay=float(tr.weight_decay),
        batch_size=int(tr.batch_size), eval_batch=int(tr.eval_batch),
        patience=int(tr.patience), max_epochs=int(tr.max_epochs),
        seed_tag=f"s{seed}",
    )


def _contrast(scores_a, scores_b, metric):
    """Paired (time_tabr vs baseline) comparison, oriented so higher=better."""
    a = orient_higher_is_better(scores_a, metric)
    b = orient_higher_is_better(scores_b, metric)
    delta = float(np.mean(a) - np.mean(b))
    p = paired_wilcoxon(a, b)
    g = hedges_g(a, b)
    return {
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "delta_oriented": delta, "p_value": p, "hedges_g": g,
        "positive": bool(delta > 0 and p < 0.05 and g >= 0.5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--n-seeds", type=int, default=25,
                    help="seeds 0..n-1 (>=25 for power, or rely on g>=0.5)")
    ap.add_argument("--splits", nargs="+", default=["temporal", "random"],
                    choices=["temporal", "random"])
    ap.add_argument("--bases", nargs="+", default=["trend", "fourier"],
                    choices=["trend", "fourier"])
    ap.add_argument("--time-mode", default="value", choices=["value", "metric", "both"])
    ap.add_argument("--archs", nargs="+", default=["mlp_t", "tabr", "time_tabr"],
                    choices=["mlp_t", "tabr", "time_tabr"])
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    seeds = list(range(args.n_seeds))
    out_dir = Path(cfg.experiment.results_dir).parent / "elec2_q2"
    out_dir.mkdir(parents=True, exist_ok=True)
    metric = metric_name("binclass")  # elec2 = roc_auc

    out_path = out_dir / f"q2_{args.time_mode}.json"
    print(f"Q2b factorial: archs={args.archs} bases={args.bases} splits={args.splits} "
          f"time_mode={args.time_mode} seeds=0..{args.n_seeds - 1}")
    print(f"results -> {out_path} (written incrementally after every seed)")

    # scores[split][basis][arch] = [per-seed test score]
    scores: dict = {sp: {b: {a: [] for a in args.archs} for b in args.bases} for sp in args.splits}
    rows = []
    summary = {"metric": metric, "n_seeds": args.n_seeds, "time_mode": args.time_mode,
               "splits": args.splits, "bases": args.bases, "complete": False,
               "contrasts": [], "rows": rows}

    def save():
        out_path.write_text(json.dumps(summary, indent=2))

    for split in args.splits:
        for basis in args.bases:
            for seed in seeds:
                seed_everything(seed)
                data = load_elec2(split=split, seed=seed)
                line = {}
                for arch in args.archs:
                    tm = args.time_mode if arch == "time_tabr" else "none"
                    r = train_timetabr(data, _cfg(cfg, arch, tm, basis, seed))
                    s = float(r["score"])
                    scores[split][basis][arch].append(s)
                    line[arch] = s
                    rows.append({"split": split, "basis": basis, "arch": arch,
                                 "time_mode": tm, "seed": seed, "score": s})
                del data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ldesc = "  ".join(f"{a}={line[a]:.4f}" for a in args.archs)
                print(f"[{split:8s}/{basis:7s} s{seed:02d}] {ldesc}")
                save()    # incremental: crash mid-run still leaves all finished rows

    # ---- pre-registered contrasts ----
    summary["complete"] = True
    print("\n==== Q2b contrasts (time_tabr vs baseline; +g => structure wins) ====")
    for split in args.splits:
        for basis in args.bases:
            sc = scores[split][basis]
            if "time_tabr" not in sc:
                continue
            for base in ("mlp_t", "tabr"):
                if base not in sc:
                    continue
                c = _contrast(sc["time_tabr"], sc[base], metric)
                c.update({"split": split, "basis": basis,
                          "candidate": "time_tabr", "baseline": base})
                summary["contrasts"].append(c)
                print(f"  [{split:8s}/{basis:7s}] time_tabr vs {base:6s}: "
                      f"delta={c['delta_oriented']:+.4f}  p={c['p_value']:.4f}  "
                      f"g={c['hedges_g']:+.2f}  positive={c['positive']}")

    # pre-registered sanity: structure gain should localize to temporal, ~0 on random
    if {"temporal", "random"} <= set(args.splits):
        print("\n  (pre-registered: gain on TEMPORAL, ~0 on RANDOM => concept exploit;")
        print("                   gain only on RANDOM => red flag, not concept.)")

    save()    # final write: now with contrasts + complete=True
    print(f"\nwrote {out_path}  <-- send me THIS one file (has all rows + contrasts)")


if __name__ == "__main__":
    main()
