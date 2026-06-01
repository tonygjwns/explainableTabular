"""Phase 0 entry point: reproduce TabM on TabReD (Cai temporal split).

Goal (PLAN.md §4 Phase 0): confirm our environment reproduces TabM within +/-1%
of the published numbers on a few small TabReD datasets, before building the
time-indexed memory + retrieval layer (Phase 1).

Usage (on the GPU machine, after SETUP.md is done):
    python scripts/run_phase0.py --config configs/tabm_baseline.yaml
    python scripts/run_phase0.py --config configs/tabm_baseline.yaml --dataset sberbank_housing

This script is the orchestration skeleton; the data loader and TabM wrapper it
calls are themselves skeletons with TODOs (need data + external/tabm).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Make `src` importable when running from repo root.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf  # noqa: E402

from src.utils.seed import seed_everything  # noqa: E402
from src.utils.metrics import compute_metric, metric_name  # noqa: E402
from src.data.tabred_loader import load_tabred, TABRED_DATASETS  # noqa: E402
from src.models.tabm_wrapper import TabMBackbone  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tabm_baseline.yaml")
    ap.add_argument("--dataset", default=None, help="run a single dataset (else all in config)")
    return ap.parse_args()


def run_single(dataset: str, cfg, seed: int) -> dict:
    """Train + evaluate TabM on one dataset for one seed. Returns a result dict."""
    seed_everything(seed)

    data = load_tabred(dataset, Path(cfg.data.root))          # TODO: needs data
    backbone = TabMBackbone(OmegaConf.to_container(cfg.model))  # TODO: needs external/tabm

    # TODO: training loop with early stopping (patience from cfg), then:
    #   y_pred = backbone.predict(data.test.X_num, data.test.X_cat)
    #   score = compute_metric(data.test.y, y_pred, data.task)
    raise NotImplementedError(
        "Wire up training loop once tabred_loader + tabm_wrapper are implemented. "
        "See PLAN.md §4 Phase 0 sub-tasks."
    )


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    datasets = [args.dataset] if args.dataset else list(cfg.data.datasets)
    results_dir = Path(cfg.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        if ds not in TABRED_DATASETS:
            raise KeyError(f"Unknown dataset {ds}; options: {list(TABRED_DATASETS)}")
        per_seed = {}
        for seed in cfg.experiment.seeds:
            res = run_single(ds, cfg, seed)
            per_seed[seed] = res
            out = results_dir / f"{ds}_seed{seed}.json"
            out.write_text(json.dumps(res, indent=2))
            print(f"[{ds} seed{seed}] {metric_name(TABRED_DATASETS[ds])} = {res.get('score')}")
        # TODO: aggregate per_seed -> mean +/- std, compare to paper (tolerance check)


if __name__ == "__main__":
    main()
