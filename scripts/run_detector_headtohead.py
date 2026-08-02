"""Do type-attributing shift tools separate rule change from label-noise decay?

The paper's closing claim is that drift-TYPE attribution without identifiability certificates is
unreliable. Every piece of evidence for it so far comes from our own instrument or our own canary
probes, and nobody uses those as detectors. This script points two established frames at the
battery cells where the ground truth is known by construction:

  DISDE-style importance-reweighting health (disde_iw_degeneration) and the covariate/within-
  overlap decomposition (covariate_shift_auc + concept_within_overlap_multi), i.e. the same
  machinery WhyShift-style analyses use to call a shift "Y|X" rather than "X".

What is and is not being claimed. Under the field-standard definition (Webb et al. 2016; Gama
et al. 2014) real drift is ANY change in P(y|x), so label-noise decay -- which changes the
conditional VARIANCE while leaving the conditional MEAN fixed -- is a TRUE positive for those
frames, not an error. The claim tested here is narrower and is about consequences, not
correctness: a tool that answers "Y|X shifted" does not distinguish a changed rule from decaying
noise, and the action its output implies (retrain on recent data, discard old labels) pays off in
the first case and not the second. Our battery makes that distinction; we check here whether the
established frames do.

Cells (from the pre-registered battery, ground truth by construction):
  reg_concept        rule genuinely rotates                      -> Y|X SHOULD be flagged
  reg_early_noisy    rule FIXED, early labels noisier            -> flagged = the ambiguity
  reg_xdep_noise     rule FIXED, x-dependent noise decay         -> flagged = the ambiguity
  reg_stable         rule fixed, no noise drift                  -> nothing should fire
  covariate          P(x) moves, rule fixed                      -> X-shift, not Y|X
  concept            rule rotates (binclass)                     -> Y|X SHOULD be flagged
A frame that fires identically on reg_concept and reg_early_noisy has not separated them.

    python scripts/run_detector_headtohead.py
    python scripts/run_detector_headtohead.py --n-seeds 5

Model-light (sklearn only). Reuses the battery's own generators so the cells are byte-for-byte
the ones the instrument was validated on -- it imports _synth from the probe rather than
re-implementing it, and does not touch the probe.
"""
from __future__ import annotations

import argparse
import importlib.util as _u
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.drift_measure import (  # noqa: E402
    covariate_shift_auc, disde_iw_degeneration, concept_within_overlap_multi,
)

CELLS = ("concept", "reg_concept", "reg_early_noisy", "reg_xdep_noise", "reg_stable", "covariate")
TRUE_RULE_CHANGE = {"concept", "reg_concept"}          # ground truth: the rule actually moved
FIXED_RULE = {"reg_early_noisy", "reg_xdep_noise", "reg_stable", "covariate"}


def _probe():
    spec = _u.spec_from_file_location(
        "_rdd", str(Path(__file__).resolve().parent / "run_deployment_decay.py"))
    m = _u.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--band", type=float, nargs=2, default=(0.1, 0.9))
    ap.add_argument("--min-per-half", type=int, default=200)
    args = ap.parse_args()

    rdd = _probe()
    out_dir = Path("results/phase1/detector_h2h"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print("\n==== type-attributing frames on ground-truth battery cells ====")
    print("  ground truth: concept/reg_concept = rule MOVED; the rest = rule FIXED\n")
    hdr = ("%-20s %-9s %8s %9s %9s %10s %10s %10s" %
           ("cell", "truth", "cov_AUC", "ESS%", "CV(w)", "gap_prim", "gap_brier", "gap_kl"))
    print(hdr); print("-" * len(hdr))
    print("  (gap_prim = auc for binclass, rmse for regression; brier/logloss/kl are "
          "classification-only)")
    for kind in CELLS:
        X, y, t, task = rdd._synth(kind)
        med = float(np.median(t)); em, lm = t <= med, t > med
        Xe, ye, Xl, yl = X[em], y[em], X[lm], y[lm]
        truth = "RULE" if kind in TRUE_RULE_CHANGE else "fixed"
        cov = covariate_shift_auc(Xe, Xl)
        try:
            iw = disde_iw_degeneration(Xe, Xl)
        except Exception as e:
            iw = {"ess_pct": None, "cv": None, "error": f"{type(e).__name__}: {e}"}
        gaps = {}
        for s in range(args.n_seeds):
            r = concept_within_overlap_multi(Xe, ye, Xl, yl, task, seed=s,
                                             band=tuple(args.band),
                                             min_per_half=args.min_per_half)
            if not r.get("measurable"):
                continue
            for k, v in (r.get("gaps") or {}).items():
                gaps.setdefault(k, []).append(v)
        mean = lambda k: (float(np.mean(gaps[k])) if gaps.get(k) else None)
        g_primary = mean("rmse") if task == "regression" else mean("auc")
        row = {"cell": f"synth_{kind}", "task": task, "ground_truth_rule_change": truth == "RULE",
               "cov_auc": cov.get("auc"), "ess_pct": iw.get("ess_pct"), "cv_w": iw.get("cv"),
               "gap_primary": g_primary, "gap_brier": mean("brier"),
               "gap_logloss": mean("logloss"), "gap_kl": mean("kl_late_early"),
               "n_measurable_seeds": len(gaps.get("brier", []))}
        rows.append(row)
        f = lambda v: ("%9.4f" % v) if isinstance(v, (int, float)) else "        -"
        print("%-20s %-9s %8s %9s %9s %10s %10s %10s" % (
            row["cell"][:20], truth,
            ("%.3f" % cov["auc"]) if cov.get("auc") is not None else "-",
            ("%.1f" % iw["ess_pct"]) if iw.get("ess_pct") is not None else "-",
            ("%.2f" % iw["cv"]) if iw.get("cv") is not None else "-",
            f(g_primary).strip(), f(row["gap_brier"]).strip(), f(row["gap_kl"]).strip()))

    print("\n  READ: a frame SEPARATES the two mechanisms only if its Y|X-side number is")
    print("  materially larger on the RULE cells than on the fixed-rule noise-decay cells.")
    print("  If reg_concept and reg_early_noisy look alike, the frame answers 'Y|X shifted'")
    print("  for both and cannot tell a practitioner which action pays off.")
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        git = ""
    blob = {"meta": {"utc": datetime.now(timezone.utc).isoformat(), "git": git,
                     "python": sys.version.split()[0], "numpy": np.__version__,
                     "argv": sys.argv[1:]}, "rows": rows}
    p = out_dir / f"detector_h2h_{git or 'nogit'}.json"
    p.write_text(json.dumps(blob, indent=2, default=float))
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
