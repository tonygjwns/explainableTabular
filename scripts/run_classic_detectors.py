"""Classic stream drift detectors on the 14-cell ground-truth battery.

WHY THIS EXISTS
  Section 2 of the manuscript currently *argues* that loss-stream detectors are type-blind by
  construction: they answer "did anything change?", not "did the rule change?", so their fires do
  not bear on drift-type attribution. This script turns that argument into a measurement. The
  battery cells have ground truth fixed by construction, including two cells where the rule is
  provably fixed and only label noise decays -- the channel the paper is about.

  It does NOT touch the instrument. `_synth` is imported from run_deployment_decay.py and nothing
  in that module is modified, so PREREG section 4's battery re-gate is not triggered.

PROTOCOL (fixed here, before execution)
  Per cell: sort rows by t, then run a prequential (test-then-train) river learner over the stream
  and feed its error signal to each detector.
    binclass   -> HoeffdingTreeClassifier, error indicator in {0,1}
    regression -> LinearRegression, absolute error (z-scaled by the first 1000 errors)
  Detectors: ADWIN and KSWIN on the numeric error stream; DDM and PageHinkley additionally on
  binclass, where a 0/1 error stream is what they were designed for.
  Recorded per (cell, detector): number of detections, index of first detection, detections per
  1000 samples. The learners are deterministic, so the 3 seeds vary the generated data (and KSWIN's
  sampling); results are reported as the mean over seeds.

PREDICTIONS (committed before the first run -- read against the outcome afterwards, including
where they fail; this is the same discipline as run_day4.sh)
  P1  Detectors fire on the label-noise-decay cells (reg_early_noisy, reg_xdep_noise) at a rate
      comparable to the true rule-change cells, i.e. within 3x.                            60%
  P2  Pure stationary cells (stable, reg_stable) fire at a strictly lower rate than both.   75%
  P3  ASYMMETRY, and this is the interesting one: DDM is one-sided by design -- it warns on error
      *increase* -- while label-noise decay makes the error stream go *down*. So DDM stays quiet
      on the noise cells while ADWIN/KSWIN (two-sided) fire on them.                        55%

HOW TO READ IT (fixed before execution)
  The claim being tested is NOT "these detectors are bad". It is that a fire carries no type
  information. So the read is the CONTRAST between rows, never a single row:
    - noise cells fire comparably to rule cells  -> section 2's argument becomes a measurement,
      and section 3.4's "sign does not separate" gains a second, weaker-instrument analogue.
    - noise cells stay quiet while rule cells fire -> the argument is WEAKER than stated, some
      classic detectors are accidentally robust to this channel, and the manuscript must say so.
      In that case P3 also tells us why, and the honest framing becomes "one-sided detectors miss
      the channel *and* miss any rule change that makes the task easier".
  Either outcome is reportable. Nothing here changes a threshold, a cascade, or a map verdict:
  these detectors emit no drift-type verdict, so they cannot enter the map.

    python scripts/run_classic_detectors.py                 # all cells, 3 seeds
    python scripts/run_classic_detectors.py --cells concept reg_early_noisy --n-seeds 1
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

RULE_CELLS = ("concept", "reg_concept", "nuisance_proxy", "concept_noise")
NOISE_CELLS = ("reg_early_noisy", "reg_xdep_noise")
NULL_CELLS = ("stable", "reg_stable", "covariate", "reg_late_noisy")
DEFAULT_CELLS = RULE_CELLS + NOISE_CELLS + NULL_CELLS


def error_stream(X, y, t, task):
    """Prequential error of a river-native incremental learner, in timestamp order.

    The learners are deterministic, so seed variation enters through the data (`_synth(seed=s)`)
    and through KSWIN's own sampling -- not through the learner.
    """
    from river import linear_model, preprocessing, tree

    order = np.argsort(t)
    X, y = X[order], y[order]
    names = [f"x{i}" for i in range(X.shape[1])]
    if task == "binclass":
        model = preprocessing.StandardScaler() | tree.HoeffdingTreeClassifier()
        errs = np.empty(len(y), dtype=float)
        for i, (xi, yi) in enumerate(zip(X, y)):
            d = dict(zip(names, xi))
            p = model.predict_one(d)
            errs[i] = 0.0 if p is None else float(p != yi)
            model.learn_one(d, yi)
        return errs
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    errs = np.empty(len(y), dtype=float)
    for i, (xi, yi) in enumerate(zip(X, y)):
        d = dict(zip(names, xi))
        errs[i] = abs(float(model.predict_one(d)) - float(yi))
        model.learn_one(d, float(yi))
    scale = np.median(errs[:1000]) or 1.0
    return errs / scale


def detect(errs, task, seed):
    """Run each applicable detector over the error stream; count detections."""
    from river import drift
    from river.drift import binary

    dets = {"ADWIN": drift.ADWIN(), "KSWIN": drift.KSWIN(seed=seed),
            "PageHinkley": drift.PageHinkley()}
    if task == "binclass":                       # 0/1 error stream: DDM's designed input
        dets["DDM"] = binary.DDM()
    out = {}
    for name, det in dets.items():
        hits = []
        for i, e in enumerate(errs):
            det.update(int(e) if name == "DDM" else float(e))
            if det.drift_detected:
                hits.append(i)
        out[name] = {"n_detections": len(hits), "first": hits[0] if hits else None,
                     "per_1k": round(1000.0 * len(hits) / len(errs), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=list(DEFAULT_CELLS))
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--out", default="results/phase1/classic_detectors")
    args = ap.parse_args()

    from run_deployment_decay import _synth                     # instrument left unmodified

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for ci, kind in enumerate(args.cells, 1):
        per_seed = []
        try:
            for s in range(args.n_seeds):
                X, y, t, task = _synth(kind, seed=s)
                per_seed.append(detect(error_stream(X, y, t, task), task, seed=s))
        except Exception as e:                                   # unknown cell name -> say so
            print(f"  [{ci}/{len(args.cells)}] {kind}: SKIP ({e})", flush=True)
            continue
        agg = {}
        for name in per_seed[0]:
            agg[name] = {
                "per_1k": round(float(np.mean([p[name]["per_1k"] for p in per_seed])), 3),
                "n_detections": round(float(np.mean([p[name]["n_detections"] for p in per_seed])), 2),
                "first": None if all(p[name]["first"] is None for p in per_seed) else
                         int(np.mean([p[name]["first"] for p in per_seed if p[name]["first"] is not None])),
            }
        truth = ("RULE MOVED" if kind in RULE_CELLS else
                 "rule fixed, noise decays" if kind in NOISE_CELLS else "rule fixed")
        rows.append({"dataset": f"synth_{kind}", "truth": truth, "task": task,
                     "n": int(len(y)), "detectors": agg, "per_seed": per_seed})
        line = "  ".join(f"{n}={agg[n]['per_1k']:.3f}/1k" for n in sorted(agg))
        print(f"  [{ci}/{len(args.cells)}] {kind:18s} [{truth:24s}] {line}", flush=True)

    blob = {"meta": {"argv": sys.argv, "n_seeds": args.n_seeds,
                     "protocol": "prequential error stream -> river detectors; instrument untouched",
                     "wall_s": round(time.time() - t0, 1)},
            "rows": rows}
    (out_dir / "classic_detectors.json").write_text(json.dumps(blob, indent=2, default=float))
    print(f"\n  wrote {out_dir}/classic_detectors.json")
    print("  READ: contrast the rule rows against the noise rows (section 2 / 3.4). A detector that")
    print("        fires on both carries no type information; one that fires only on rule change is")
    print("        one-sided and will also miss any rule change that makes the task easier.")


if __name__ == "__main__":
    main()
