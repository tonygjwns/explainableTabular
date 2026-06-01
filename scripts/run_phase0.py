"""Phase 0 entry point: reproduce TabM on TabReD (Cai temporal split).

Goal (PLAN.md §4 Phase 0): confirm our environment reproduces TabM within +/-1%
of the published numbers on a few small TabReD datasets, before building the
time-indexed memory + retrieval layer (Phase 1).

Usage (on the GPU machine, after SETUP.md is done):
    python scripts/run_phase0.py --config configs/tabm_baseline.yaml
    python scripts/run_phase0.py --config configs/tabm_baseline.yaml --dataset sberbank_housing

Implemented end-to-end (loader + TabM wrapper + trainer). Requires the TabReD data
to be generated (SETUP.md §3) and the tabm package installed (pip install -e external/tabm).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Make `src` importable when running from repo root.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.utils.seed import seed_everything  # noqa: E402
from src.utils.metrics import metric_name  # noqa: E402
from src.data.tabred_loader import load_tabred, TABRED_DATASETS, DIRNAME  # noqa: E402
from src.training.trainer import TrainConfig, train_tabm_baseline  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tabm_baseline.yaml")
    ap.add_argument("--dataset", default=None, help="run a single dataset (else all in config)")
    return ap.parse_args()


def run_single(dataset: str, cfg, seed: int) -> dict:
    """Train + evaluate TabM on one dataset for one seed."""
    seed_everything(seed)
    data = load_tabred(dataset, Path(cfg.data.root), split=cfg.experiment.split)
    tcfg = TrainConfig(
        k=cfg.model.k, n_blocks=cfg.model.n_blocks, d_block=cfg.model.d_block,
        dropout=cfg.model.dropout,
        lr=cfg.training.learning_rate if "learning_rate" in cfg.training else 2e-3,
        weight_decay=cfg.training.weight_decay if "weight_decay" in cfg.training else 0.0,
        batch_size=cfg.training.batch_size, patience=cfg.training.patience,
        max_epochs=cfg.training.max_epochs, seed_tag=f"s{seed}",
    )
    res = train_tabm_baseline(data, tcfg)
    res["seed"] = seed
    return res


def data_present(dataset: str, root: Path) -> bool:
    """True if the dataset's preprocessed output exists (so we can skip missing ones)."""
    base = Path(root) / DIRNAME[dataset]
    return (base / "X_meta.npy").exists() and (base / "Y.npy").exists()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    datasets = [args.dataset] if args.dataset else list(cfg.data.datasets)
    root = Path(cfg.data.root)
    results_dir = Path(cfg.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    done, skipped = [], []
    for ds in tqdm(datasets, desc="datasets", dynamic_ncols=True):
        if ds not in TABRED_DATASETS:
            raise KeyError(f"Unknown dataset {ds}; options: {list(TABRED_DATASETS)}")
        if not data_present(ds, root):
            print(f"[skip] {ds}: data not found under {root} — run preprocessing first.")
            skipped.append(ds)
            continue
        scores = []
        for seed in tqdm(list(cfg.experiment.seeds), desc=f"{ds} seeds", leave=False, dynamic_ncols=True):
            res = run_single(ds, cfg, seed)
            scores.append(res["score"])
            (results_dir / f"{ds}_seed{seed}.json").write_text(json.dumps(res, indent=2))
            print(f"[{ds} seed{seed}] {metric_name(TABRED_DATASETS[ds])} = {res['score']:.4f}")
        mean, std = float(np.mean(scores)), float(np.std(scores))
        summary = {"dataset": ds, "metric": metric_name(TABRED_DATASETS[ds]),
                   "seeds": list(cfg.experiment.seeds), "mean": mean, "std": std,
                   "scores": scores}
        (results_dir / f"{ds}_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{ds}] {summary['metric']} = {mean:.4f} +/- {std:.4f}  "
              f"(compare to TabM/TabReD table within {cfg.reproduction_check.tolerance:.0%})")
        done.append(ds)

    print("\n==== Phase 0 run complete ====")
    print(f"done   : {done}")
    print(f"skipped: {skipped}  (no data; preprocess then re-run)")


if __name__ == "__main__":
    main()
