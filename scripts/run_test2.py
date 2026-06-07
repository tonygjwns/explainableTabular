"""Phase-1 Sanity Test 2: extrapolation — does P_k(t) track real future drift?

PRE_REGISTRATION §3.2 / EXPERIMENT_PLAN §8. Train a time-indexed model on the
first 70% of TRAINING time, then on the held-out future 30% compare the memory's
extrapolated usage-weighted centroid C_extrap(s) to the real data centroid
C_real(s) per future time slice (see src/analysis/extrapolation.py).

PASS (PRE_REG §3.2): R^2 >= 30% in >= 2/4 datasets AND trend direction matches.
(H1b also references Pearson r >= 0.4.) With ~8 slices these are noisy — report
with the trajectory plot and judge as evidence.

    python scripts/run_test2.py --config configs/phase1.yaml --dataset sberbank_housing
    python scripts/run_test2.py --config configs/phase1.yaml   # all 4 sanity datasets
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
from src.data.tabred_loader import load_tabred, TABRED_DATASETS, TabularSplit, TabReDDataset  # noqa: E402
from src.training.phase1_trainer import train_phase1  # noqa: E402
from src.training.trainer import _prep_numeric  # noqa: E402
from src.analysis.extrapolation import mean_retrieval_weights, extrapolation_fit  # noqa: E402
# reuse the sanity config->Phase1Config builder
from run_phase1_sanity import make_cfg  # noqa: E402


def _subset(split: TabularSplit, mask: np.ndarray) -> TabularSplit:
    sel = lambda a: None if a is None else a[mask]
    return TabularSplit(X_num=sel(split.X_num), X_bin=sel(split.X_bin), X_cat=sel(split.X_cat),
                        y=split.y[mask], t=split.t[mask], t_raw=split.t_raw[mask])


def run_one(ds: str, cfg, seed: int, root: Path, frac: float, n_slices: int) -> dict:
    seed_everything(seed)
    data = load_tabred(ds, root, split=cfg.experiment.split)
    tr = data.train
    q = float(np.quantile(tr.t, frac))
    early_m = tr.t <= q
    late_m = tr.t > q
    if late_m.sum() < n_slices * 5:
        raise RuntimeError(f"{ds}: too few future rows ({int(late_m.sum())}) for {n_slices} slices")
    early, late = _subset(tr, early_m), _subset(tr, late_m)

    # train time-indexed on the early 70% (val for early-stop = original val)
    data_early = TabReDDataset(name=ds, task=data.task, split=data.split,
                               train=early, val=data.val, test=late,
                               t_min=data.t_min, t_max=data.t_max)
    pcfg = make_cfg(cfg, seed, True)        # time_indexed=True (extrapolate the drift)
    res = train_phase1(data_early, pcfg)
    model = res["model"]

    # prepped numerics (fit on early train) for both early (weights) and late (probe)
    (xnum_early, xnum_late), _ = _prep_numeric(early, late)
    wbar = mean_retrieval_weights(model, xnum_early, early.X_cat, early.t,
                                  device=next(model.parameters()).device.type)
    fit = extrapolation_fit(model, xnum_late, late.X_cat, late.t, wbar, n_slices=n_slices)
    fit.update({"dataset": ds, "seed": seed, "train_frac": frac,
                "n_future_rows": int(late_m.sum())})
    del model, data, data_early
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.7, help="train-time fraction to train on")
    ap.add_argument("--n-slices", type=int, default=8)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    root = Path(cfg.data.root)
    datasets = [args.dataset] if args.dataset else list(cfg.data.sanity_datasets)
    out_dir = Path(cfg.experiment.results_dir).parent / "test2"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in datasets:
        if ds not in TABRED_DATASETS:
            raise KeyError(f"Unknown dataset {ds}; options: {list(TABRED_DATASETS)}")
        fit = run_one(ds, cfg, args.seed, root, args.frac, args.n_slices)
        rows.append(fit)
        print(f"[{ds}] r(weighted)={fit['mean_pearson_r_weighted']:+.3f}  "
              f"R2(weighted)={fit['mean_r2_weighted']:.3f}  "
              f"dir_agree={fit['direction_agreement_frac']:.2f}  "
              f"(real move={fit['real_centroid_movement']:.3f}, "
              f"extrap move={fit['extrap_centroid_movement']:.3f})")
        (out_dir / f"{ds}_seed{args.seed}.json").write_text(json.dumps(fit, indent=2))

    # PRE_REG §3.2: R2>=30% in >=2/4 (+ direction). Report; judgment is human.
    n_r2 = sum(1 for r in rows if r["mean_r2_weighted"] >= 0.30
               and r["direction_agreement_frac"] >= 0.5)
    n_r04 = sum(1 for r in rows if r["mean_pearson_r_weighted"] >= 0.40)
    verdict = {
        "datasets": datasets, "seed": args.seed, "rows": rows,
        "n_datasets_R2>=0.30_and_dir": n_r2,
        "n_datasets_r>=0.40": n_r04,
        "pass_hint_R2": bool(n_r2 >= 2 and (not args.dataset)),
        "complete": not bool(args.dataset),
        "NOTE": "PRE_REG §3.2: R2>=30% & direction in >=2/4. Noisy with ~8 slices; "
                "judge with trajectory plot. Fixed memory gives r~0 by construction.",
    }
    (out_dir / "test2_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"\n==== Test 2 (extrapolation) ====")
    print(f"  datasets with R2>=0.30 & dir>=0.5: {n_r2}  |  with r>=0.40: {n_r04}")
    print(f"  PASS hint (R2 in >=2/4): {verdict['pass_hint_R2']}"
          + ("  (subset — diagnostic only)" if args.dataset else ""))


if __name__ == "__main__":
    main()
