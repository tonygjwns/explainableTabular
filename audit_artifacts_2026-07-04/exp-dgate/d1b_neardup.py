"""D1b: NEAR-duplicates (noise 0.02, t jittered within the same coarse day) on stationary X —
the realistic TabReD row structure (same application/offer logged multiple times, snapshot
features). Does D still cross the 0.96 gate?"""
import importlib.util
import json

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "dd", r"C:\Users\joon\Desktop\ExplainableTab\scripts\run_deployment_decay.py")
dd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dd)

rng = np.random.default_rng(1)
n_base, DIM, m = 6000, 10, 5
Xb = rng.normal(0, 1, (n_base, DIM))              # stationary
day = rng.integers(0, 45, n_base)                 # coarse day per base row (45 unique, ecom-like)
X = np.repeat(Xb, m, axis=0) + 0.02 * rng.normal(0, 1, (n_base * m, DIM))
t = np.repeat(day, m).astype(float)               # near-dups share the SAME day
o = np.argsort(t, kind="stable")
X, t = X[o], t[o]

win, Keff = dd._assign_windows(t, 10, False)
D = dd._disjointness(X, win, Keff)
sh = np.random.default_rng(0).permutation(len(t))
wsh, Ksh = dd._assign_windows(t[sh], Keff, False)
Dsh = dd._disjointness(X, wsh, Ksh)
print(f"near-dup m=5 noise=0.02 coarse-day t: D = {D:.4f}  D_shuffle = {Dsh:.4f}", flush=True)

out = {"near_dup_m5_noise02_coarseday": {"D": D, "D_shuffle": Dsh}}

# noise sweep at m=5: how much within-group noise kills the memorization inflation?
for noise in (0.05, 0.1, 0.3):
    Xn = np.repeat(Xb, m, axis=0) + noise * rng.normal(0, 1, (n_base * m, DIM))
    Xn, tn = Xn[o], t
    winn, Kn = dd._assign_windows(tn, 10, False)
    Dn = dd._disjointness(Xn, winn, Kn)
    out[f"near_dup_m5_noise{noise}"] = Dn
    print(f"near-dup m=5 noise={noise}: D = {Dn:.4f}", flush=True)

out_path = (r"C:\Users\joon\AppData\Local\Temp\claude"
            r"\C--Users-joon-Desktop-ExplainableTab"
            r"\e9542da3-1a26-4cb3-9fd3-ce63bc60e8d9\scratchpad\exp-dgate\d1b_results.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"saved {out_path}", flush=True)
