"""Close the two provenance gaps the number audit left open (RESULTS section 31).

The manuscript quotes two things that no committed JSON artifact carries. Neither is a wrong claim
-- both were executed -- but neither is reproducible from the repository, which is a defect in the
reproducibility package rather than in the result. This script re-measures both and writes
artifacts. It imports from run_deployment_decay.py without modifying it, so the PREREG section 4
battery re-gate is not triggered.

J1  Appendix B.3's size-term row for synth_reg_stable quotes absolute z-RMSE levels
    (recent -0.1343 -> recent+old -0.1222, size term +0.01212). The size term is exact in
    audit_artifacts_2026-07-04/exp-reg/reg_controls_results.json (staleness_harm -0.01212), but
    that run did not store per-seed absolute levels, so the two levels are unbacked. `assess`
    stores them under per_seed_abs, so re-running the cell recovers them.

    HONEST FRAMING, fixed before the run: this is a RE-MEASUREMENT, not a recovery of the original
    numbers. If it returns levels that differ from the quoted ones, the manuscript adopts the
    re-measured pair together with the size term computed from the SAME run -- never a level from
    one run beside a size term from another -- and RESULTS records the substitution. The claim the
    row supports (the size term is 61% of the decision floor on regression) is what must survive;
    if it does not survive, that is reportable too.

J2  Appendix C's learnability-gate row quotes the executed vacuity control: junk heavy-tailed
    top-variance features make the injected rule unlearnable (in-window AUC 0.506) while a
    learnable control on the same harness recovers (AUC 0.964, recovery +0.195). Its source is
    AUDIT_FINAL_2026-07-04.md section C1 -- a markdown writeup, no JSON. This re-runs both arms.

    Same framing: re-measurement. The row's claim is qualitative (unlearnable geometry yields a
    vacuous certificate, learnable geometry recovers), and that is what has to hold. The exact
    constants are updated to whatever the artifact says.

    python scripts/run_provenance_backfill.py            # both jobs
    python scripts/run_provenance_backfill.py --job j1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def j1(n_seeds, K):
    """Absolute z-RMSE levels for the regression size-term row."""
    from run_deployment_decay import _synth, assess

    X, y, t, task = _synth("reg_stable")
    r = assess("synth_reg_stable", X, y, t, task, K=K, n_seeds=n_seeds)
    abs_rows = [d for d in (r.get("per_seed_abs") or []) if isinstance(d, dict)]
    lv = {}
    for key in ("recent", "recent_old", "old", "recent_old_denoised"):
        vals = [d[key] for d in abs_rows if isinstance(d.get(key), (int, float))]
        if vals:
            lv[key] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                       "n": len(vals)}
    size_term = None
    if "recent" in lv and "recent_old" in lv:
        size_term = lv["recent_old"]["mean"] - lv["recent"]["mean"]
    print(f"  J1 levels: recent {lv.get('recent',{}).get('mean')!r} -> "
          f"recent_old {lv.get('recent_old',{}).get('mean')!r}   size term {size_term!r}")
    print(f"     staleness_harm from the same run: {r.get('staleness_harm')!r}")
    return {"job": "j1", "cell": "synth_reg_stable", "K": K, "n_seeds": n_seeds,
            "levels": lv, "size_term_from_levels": size_term,
            "staleness_harm": r.get("staleness_harm"),
            "staleness_harm_ci": r.get("staleness_harm_ci"),
            "quoted_in_manuscript": {"recent": -0.1343, "recent_old": -0.1222, "size_term": 0.01212},
            "row": r}


def j2(n_seeds, K, dfs=(4.0, 2.0, 1.0, 0.6)):
    """Injection vacuity: sweep the junk geometry from learnable to unlearnable.

    A first attempt planted the rule in a single heavy-tailed geometry and got a LEARNABLE result
    (score 0.858), not the unlearnable one Appendix C quotes. Tuning that construction until it hit
    0.506 would be fitting the control to its target, so instead the tail weight is swept and the
    whole curve reported. That is a stronger backfill than the single point: it shows learnability
    crossing the pre-registered 0.65 gate as the geometry degrades, and the certificate going
    vacuous with it, rather than asserting one number.

    Calls the injection helpers directly rather than through `assess`, because the cascade only
    reaches the injection stage for cells it classes unidentifiable. The instrument is unmodified.
    """
    from run_deployment_decay import LEARN_AUC, _injection_recovers, _synth

    X, y, t, task = _synth("stable")
    arms, rng = [], np.random.default_rng(0)
    for label, Xa in [("learnable (cell geometry)", X)] + [
            (f"junk t(df={d})", np.hstack([rng.standard_t(df=d, size=(len(X), 2)) * 12.0, X]))
            for d in dfs]:
        rec, mean, ci, learnable, lscore, feats, per_seed = _injection_recovers(
            Xa, t, task, K=K, by_value=False, max_train=6000, n_seeds=n_seeds)
        arms.append({"arm": label, "learn_score": lscore, "learnable": bool(learnable),
                     "gate": LEARN_AUC, "injected_staleness": mean, "ci": ci,
                     "recovered": bool(rec), "per_seed": per_seed})
        print(f"  J2 {label:24s} learn={lscore:.3f} {'LEARNABLE' if learnable else 'VACUOUS  '} "
              f"inj={mean:+.4f} recovered={bool(rec)}")
    return {"job": "j2", "K": K, "n_seeds": n_seeds, "arms": arms,
            "read": "learnability crossing the gate is the claim; the certificate is vacuous below it",
            "quoted_in_manuscript": {"junk_auc": 0.506, "learnable_auc": 0.964,
                                     "learnable_recovery": 0.195}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", choices=["j1", "j2", "both"], default="both")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--windows", type=int, default=10)
    ap.add_argument("--out", default="results/phase1/provenance_backfill")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, results = time.time(), []
    if args.job in ("j1", "both"):
        results.append(j1(args.n_seeds, args.windows))
    if args.job in ("j2", "both"):
        results.append(j2(args.n_seeds, args.windows))
    blob = {"meta": {"argv": sys.argv, "wall_s": round(time.time() - t0, 1),
                     "purpose": "backfill artifacts for two manuscript numbers with no committed JSON"},
            "rows": results}
    (out_dir / "provenance_backfill.json").write_text(json.dumps(blob, indent=2, default=float))
    print(f"\n  wrote {out_dir}/provenance_backfill.json")
    print("  READ: adopt the re-measured numbers wherever they differ, keeping levels and size term")
    print("        from the same run, and record the substitution in RESULTS.")


if __name__ == "__main__":
    main()
