"""Phase-1 Sanity Test 3: retrieval meaningfulness + prototype-trajectory viz.

PRE_REGISTRATION §3.3 (non-gating, qualitative PASS by peer+self). Trains one
time-indexed Phase1Model on a dataset, then produces the quantitative aids and
the trajectory plot the human judgment is made on:
  - retrieval concentration on the TEST split (participation ratio / entropy /
    top-k mass) — is retrieval meaningfully concentrated, not uniform?
  - prototype trajectories P_k(t) over the time grid -> PCA/UMAP plot + geometry
    (path length, straightness) — smooth & directional?
  - label summary of the biggest-moving prototypes — interpretable?

    python scripts/run_test3.py --config configs/phase1.yaml --dataset sberbank_housing

Outputs to results/phase1/test3/<dataset>/: metrics.json + trajectories_*.png.
Heuristic readouts are aids only; the actual PASS is the peer+self call.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.utils.seed import seed_everything  # noqa: E402
from src.utils.metrics import metric_name  # noqa: E402
from src.data.tabred_loader import load_tabred, TABRED_DATASETS  # noqa: E402
from src.training.phase1_trainer import Phase1Config, train_phase1  # noqa: E402
from src.training.trainer import _prep_numeric  # noqa: E402
from src.analysis.retrieval_trajectory import (  # noqa: E402
    retrieval_concentration, prototype_trajectories, trajectory_metrics, plot_trajectories,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--dataset", default="sberbank_housing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-times", type=int, default=50)
    ap.add_argument("--method", default="pca", choices=["pca", "umap"])
    ap.add_argument("--n-proto", type=int, default=20, help="# top-mover prototypes to plot")
    ap.add_argument("--tau", type=float, default=None,
                    help="override retrieval temperature (sweep for concentration)")
    ap.add_argument("--lambda-smooth", type=float, default=None,
                    help="override L_smooth weight (set 0 to test if drift can move)")
    ap.add_argument("--n-harmonics", type=int, default=None,
                    help="override Fourier harmonics (lower => smoother/more directional drift)")
    ap.add_argument("--diag-every", type=int, default=5,
                    help="print internal training diagnostics every N epochs (0=off)")
    return ap.parse_args()


def make_cfg(cfg, seed: int, tau: float = None, lambda_smooth: float = None,
             n_harmonics: int = None) -> Phase1Config:
    m, mem, tr = cfg.model, cfg.memory, cfg.training
    return Phase1Config(
        k=m.k, n_blocks=m.n_blocks, d_block=m.d_block, dropout=m.dropout,
        n_prototypes=mem.n_prototypes, rank=mem.rank, mem_hidden=mem.mem_hidden,
        tau_temp=(mem.tau_temp if tau is None else tau), predictor_hidden=mem.predictor_hidden,
        time_indexed=True, inject_time_input=mem.inject_time_input,
        input_time_out_dim=mem.input_time_out_dim, mem_time_out_dim=mem.mem_time_out_dim,
        n_harmonics=(mem.n_harmonics if n_harmonics is None else n_harmonics),
        time_periods=tuple(mem.time_periods),
        kmeans_init=mem.kmeans_init, n_slices=mem.n_slices,
        kmeans_max_samples=mem.kmeans_max_samples,
        lambda_smooth=(mem.lambda_smooth if lambda_smooth is None else lambda_smooth),
        lr=tr.learning_rate, weight_decay=tr.weight_decay, batch_size=tr.batch_size,
        eval_batch=tr.eval_batch, patience=tr.patience, max_epochs=tr.max_epochs,
        seed_tag=f"t3s{seed}",
    )


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    ds = args.dataset
    if ds not in TABRED_DATASETS:
        raise KeyError(f"Unknown dataset {ds}; options: {list(TABRED_DATASETS)}")
    out_dir = Path(cfg.experiment.results_dir).parent / "test3" / ds
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- train one time-indexed model ----
    seed_everything(args.seed)
    data = load_tabred(ds, Path(cfg.data.root), split=cfg.experiment.split)
    pcfg = make_cfg(cfg, args.seed, tau=args.tau, lambda_smooth=args.lambda_smooth,
                    n_harmonics=args.n_harmonics)
    pcfg.diag_every = args.diag_every
    res = train_phase1(data, pcfg)
    model = res["model"]
    metric = metric_name(data.task)
    tau_used = args.tau if args.tau is not None else float(cfg.memory.tau_temp)
    lam_used = args.lambda_smooth if args.lambda_smooth is not None else float(cfg.memory.lambda_smooth)
    nharm_used = args.n_harmonics if args.n_harmonics is not None else int(cfg.memory.n_harmonics)
    print(f"[{ds}] trained time-indexed model (tau={tau_used}, lambda_smooth={lam_used}, "
          f"n_harmonics={nharm_used}): {metric}={res['score']:.4f}")

    # ---- retrieval concentration on TEST split ----
    (xnum_tr, xnum_te), _ = _prep_numeric(data.train, data.test)
    conc = retrieval_concentration(model, xnum_te, data.test.X_cat, data.test.t)
    print(f"  retrieval: PR={conc['participation_ratio_mean']:.1f}/{conc['K']} "
          f"(frac={conc['participation_ratio_frac']:.3f}), "
          f"entropy_frac={conc['entropy_frac']:.3f}, top5_mass={conc['top5_mass_mean']:.3f}")
    print(f"  dist diag: sqdist_mean={conc['sqdist_mean']:.3f}, "
          f"across-proto rel_spread={conc['sqdist_rel_spread']:.4f}  "
          f"(rel_spread~0 => prototypes indistinguishable from z; tau won't help)")

    # ---- trajectories + geometry ----
    tg, P = prototype_trajectories(model, n_times=args.n_times)
    tm = trajectory_metrics(P)
    print(f"  trajectory: straightness median(all)={tm['straightness_median']:.3f}, "
          f"MOVERS(top {tm['n_movers_considered']})={tm['mover_straightness_median']:.3f} "
          f"(1=straight,0=wiggly), mover path_len median={tm['mover_path_len_median']:.3f}")

    tag = f"{args.method}_tau{tau_used}_lam{lam_used}_h{nharm_used}"
    png = plot_trajectories(P, tg, str(out_dir / f"trajectories_{tag}.png"),
                            n_proto=args.n_proto, mover_idx=tm["movers_idx"], method=args.method)
    print(f"  saved plot -> {png}")

    # ---- label summary of biggest movers (interpretability) ----
    label_sum = model.value_module.prototype_label_summary().cpu().numpy()
    movers = tm["movers_idx"]
    movers_labels = label_sum[movers].tolist()

    # heuristic readouts (AIDS ONLY — the real PASS is the peer+self judgment)
    aid = {
        "retrieval_concentrated_hint": bool(conc["participation_ratio_frac"] < 0.10),
        "trajectory_directional_hint": bool(tm["mover_straightness_median"] > 0.5),
    }

    payload = {
        "dataset": ds, "task": data.task, "metric": metric, "score": res["score"],
        "seed": args.seed, "tau": tau_used, "lambda_smooth": lam_used,
        "n_times": args.n_times, "method": args.method,
        "retrieval_concentration": conc,
        "trajectory": {k: v for k, v in tm.items() if not k.startswith("_")},
        "biggest_movers_idx": movers,
        "biggest_movers_label_summary": movers_labels,
        "heuristic_aids": aid,
        "plot": str(png),
        "NOTE": "Test 3 PASS is a qualitative peer+self judgment; hints are aids only.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out_dir/'metrics.json'}")
    print(f"Hints (NOT a verdict): concentrated={aid['retrieval_concentrated_hint']}, "
          f"directional={aid['trajectory_directional_hint']}")


if __name__ == "__main__":
    main()
