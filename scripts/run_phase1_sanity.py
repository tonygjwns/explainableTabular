"""Phase-1 Sanity Check — Test 1 (gating): time-indexed memory vs fixed memory.

EXPERIMENT_PLAN.md §8 / PRE_REGISTRATION.md §3.1. For each sanity dataset and
seed, train the SAME Phase1Model twice -- once with the time-indexed prototype
memory P_k(t)=P_k^base+drift_k(Fourier(t)) and once with drift disabled
(fixed memory P_k=P_k^base) -- and compare with paired Wilcoxon + BH-FDR +
Hedges' g (src/utils/stats.py).

PASS (pre-registered): time-indexed > fixed, post-FDR p<alpha AND Hedges' g >=
0.3, in >= min_datasets of the 4; and never significantly worse.

    python scripts/run_phase1_sanity.py --config configs/phase1.yaml
    python scripts/run_phase1_sanity.py --config configs/phase1.yaml --dataset sberbank_housing

NOTE (scope): this implements the Test-1 GATE (drift on/off on the TabM backbone).
PRE_REGISTRATION §2.1 also envisions a Cai-modulation-augmented backbone as the
fixed context; that Cai baseline is a separate axis still to be added (§7.2). The
gating comparison itself — does time-indexing the memory help — is what runs here.
Tests 2-4 (extrapolation / retrieval+trajectory / time-injection) live in
src/analysis/ and are added next.

Runtime warning: 4 datasets x 5 seeds x 2 variants = 40 trainings; homecredit is
large (~10 min each). Use --dataset to run a subset, and nohup for the full run.
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
from src.utils.metrics import metric_name  # noqa: E402
from src.data.tabred_loader import load_tabred, TABRED_DATASETS  # noqa: E402
from src.training.phase1_trainer import Phase1Config, train_phase1  # noqa: E402
from src.utils.stats import compare_across_datasets, passes_test1  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--dataset", default=None, help="run a single sanity dataset")
    ap.add_argument("--predictor-mode", default=None,
                    choices=["concat", "memory_only", "residual"],
                    help="override predictor mode (force memory engagement)")
    return ap.parse_args()


def make_cfg(cfg, seed: int, time_indexed: bool) -> Phase1Config:
    m, mem, tr = cfg.model, cfg.memory, cfg.training
    return Phase1Config(
        k=m.k, n_blocks=m.n_blocks, d_block=m.d_block, dropout=m.dropout,
        n_prototypes=mem.n_prototypes, rank=mem.rank, mem_hidden=mem.mem_hidden,
        tau_temp=mem.tau_temp, predictor_hidden=mem.predictor_hidden,
        predictor_mode=mem.predictor_mode,
        time_indexed=time_indexed, inject_time_input=mem.inject_time_input,
        input_time_out_dim=mem.input_time_out_dim, mem_time_out_dim=mem.mem_time_out_dim,
        n_harmonics=mem.n_harmonics, time_periods=tuple(mem.time_periods),
        kmeans_init=mem.kmeans_init, n_slices=mem.n_slices,
        kmeans_max_samples=mem.kmeans_max_samples, lambda_smooth=mem.lambda_smooth,
        lr=tr.learning_rate, weight_decay=tr.weight_decay, batch_size=tr.batch_size,
        eval_batch=tr.eval_batch, patience=tr.patience, max_epochs=tr.max_epochs,
        seed_tag=f"s{seed}{'T' if time_indexed else 'F'}",
    )


def run_one(dataset: str, cfg, seed: int, time_indexed: bool, root: Path) -> float:
    """Train one (dataset, seed, variant) and return the test score."""
    seed_everything(seed)
    data = load_tabred(dataset, root, split=cfg.experiment.split)
    res = train_phase1(data, make_cfg(cfg, seed, time_indexed))
    score = float(res["score"])
    # free GPU memory between the 40 runs
    res.pop("model", None)
    del data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.predictor_mode:
        cfg.memory.predictor_mode = args.predictor_mode
    root = Path(cfg.data.root)
    seeds = list(cfg.experiment.seeds)
    datasets = [args.dataset] if args.dataset else list(cfg.data.sanity_datasets)
    results_dir = Path(cfg.experiment.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    inject = bool(cfg.memory.inject_time_input)
    print(f"Test 1 contrast: time_indexed=True vs False, predictor_mode={cfg.memory.predictor_mode}, "
          f"inject_time_input={inject} (BOTH).")
    print("  inject=False => isolates memory-time (drift on/off only); "
          "inject=True => memory-time ON TOP OF input-time (stricter).")

    scores_time: dict[str, list[float]] = {}
    scores_fixed: dict[str, list[float]] = {}
    metric_by_dataset: dict[str, str] = {}

    for ds in datasets:
        if ds not in TABRED_DATASETS:
            raise KeyError(f"Unknown dataset {ds}; options: {list(TABRED_DATASETS)}")
        metric_by_dataset[ds] = metric_name(TABRED_DATASETS[ds])
        st, sf = [], []
        for seed in seeds:
            s_time = run_one(ds, cfg, seed, True, root)
            s_fix = run_one(ds, cfg, seed, False, root)
            st.append(s_time); sf.append(s_fix)
            rec = {"dataset": ds, "seed": seed, "metric": metric_by_dataset[ds],
                   "time_indexed": s_time, "fixed": s_fix}
            (results_dir / f"{ds}_seed{seed}.json").write_text(json.dumps(rec, indent=2))
            print(f"[{ds} s{seed}] {metric_by_dataset[ds]}: "
                  f"time={s_time:.4f}  fixed={s_fix:.4f}  diff={s_time - s_fix:+.4f}")
        scores_time[ds] = st
        scores_fixed[ds] = sf
        (results_dir / f"{ds}_summary.json").write_text(json.dumps(
            {"dataset": ds, "metric": metric_by_dataset[ds],
             "time_indexed": st, "fixed": sf,
             "mean_time": float(np.mean(st)), "mean_fixed": float(np.mean(sf))}, indent=2))

    # ---- Test 1 verdict (only meaningful with all sanity datasets present) ----
    t1 = cfg.test1
    results, reject = compare_across_datasets(
        scores_time, scores_fixed, metric_by_dataset, alpha=t1.alpha)
    verdict = passes_test1(results, reject,
                           min_datasets=t1.min_datasets, min_hedges_g=t1.min_hedges_g)

    print("\n==== Test 1: time-indexed vs fixed memory ====")
    rows = []
    for res, rej in zip(results, reject):
        # delta/g are on oriented scores (higher = better) regardless of metric
        print(f"  {res.dataset:20s} delta={res.delta:+.4f}  p={res.p_value:.4f}  "
              f"g={res.hedges_g:+.2f}  reject(FDR)={bool(rej)}")
        rows.append({"dataset": res.dataset, "delta_oriented": res.delta,
                     "p_value": res.p_value, "hedges_g": res.hedges_g,
                     "reject_fdr": bool(rej)})
    print(f"\nTest 1 ({'all 4' if not args.dataset else 'SUBSET — not a verdict'}): "
          f"{'PASS' if verdict else 'FAIL/AMBIGUOUS'}  "
          f"(need time>fixed post-FDR & g>={t1.min_hedges_g} in >={t1.min_datasets}/4)")
    if args.dataset:
        print("  (ran a single dataset; full PASS/FAIL needs all 4 — this is diagnostic only)")

    (results_dir / "test1_verdict.json").write_text(json.dumps(
        {"datasets": datasets, "seeds": seeds, "rows": rows,
         "contrast": f"time_indexed True vs False; inject_time_input={inject} (both)",
         "inject_time_input": inject,
         "min_datasets": t1.min_datasets, "min_hedges_g": t1.min_hedges_g,
         "alpha": t1.alpha, "pass": bool(verdict),
         "complete": not bool(args.dataset)}, indent=2))


if __name__ == "__main__":
    main()
