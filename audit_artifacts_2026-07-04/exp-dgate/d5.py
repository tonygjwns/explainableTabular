"""D5: staleness under TRUE covariate disjointness (their own covariate synth, X0 += 6t,
fixed nonlinear rule). Full assess() with 10 seeds, exactly as the real-data runs.
Questions:
 1. Is staleness significantly NEGATIVE (old data HELPS) or neutral under true disjointness?
    (Calibrates what homesite's stale=-0.0039 CI[-.0045,-.0034] at D=1.000 implies.)
 2. Does the injection control RECOVER (inj_stale fires) despite true disjointness?
    (Calibrates whether INJECTION-RECOVERED at D=1.000 on cooking/delivery proves overlap.)
"""
import importlib.util
import json

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "dd", r"C:\Users\joon\Desktop\ExplainableTab\scripts\run_deployment_decay.py")
dd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dd)

X, y, t, task = dd._synth("covariate")
r = dd.assess("synth_covariate_10seed", X, y, t, task, K=10, n_seeds=10, max_train=6000)
print(json.dumps({k: r[k] for k in
                  ("dataset", "D_strip", "D_full", "D_shuffle", "staleness_harm",
                   "staleness_harm_ci", "recency_gain", "recency_gain_ci", "decay",
                   "decay_ci", "injected_staleness", "delta_staleness", "n_proxy_stripped",
                   "verdict", "trust")}, indent=2, default=float), flush=True)

# stronger-separation variant: X0 += 12t (old and future windows fully non-overlapping in X0)
X2, y2, t2, _ = dd._synth("covariate")
X2 = X2.copy()
X2[:, 0] += 6.0 * t2   # already has +6t from _synth; add another +6t -> total +12t
r2 = dd.assess("synth_covariate_12t", X2, y2, t2, "binclass", K=10, n_seeds=10, max_train=6000)
print(json.dumps({k: r2[k] for k in
                  ("dataset", "D_strip", "staleness_harm", "staleness_harm_ci",
                   "injected_staleness", "verdict", "trust")}, indent=2, default=float),
      flush=True)

out_path = (r"C:\Users\joon\AppData\Local\Temp\claude"
            r"\C--Users-joon-Desktop-ExplainableTab"
            r"\e9542da3-1a26-4cb3-9fd3-ce63bc60e8d9\scratchpad\exp-dgate\d5_results.json")
with open(out_path, "w") as f:
    json.dump({"covariate_6t": r, "covariate_12t": r2}, f, indent=2, default=float)
print(f"saved {out_path}", flush=True)
