"""Adversarial audit of the identifiability gate D (_disjointness) in
scripts/run_deployment_decay.py.

D1  duplicates on stationary synth        -> does row memorization inflate D?
D2  entity recurrence, random times       -> D ~ 0.5 expected (entity in many windows)
D2b entity cohorts (one narrow time band) -> does entity memorization = window memorization
                                             inflate D despite iid latents (no covariate shift)?
D3  mild mean drift (0.5*t on 2 feats)    -> cohort vs random-time vs pure-iid comparison
D4  time-shuffle control mechanics        -> does D_shuffle stay ~0.5 on cohort/dup data
                                             (i.e. does the d-gate-suspect flag actually catch
                                             memorization)? plus skewed-window-size variant.
All D values printed are actual _disjointness outputs from the repo module.
"""
import importlib.util
import json
import sys

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "dd", r"C:\Users\joon\Desktop\ExplainableTab\scripts\run_deployment_decay.py")
dd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dd)

OUT = {}


def D_of(X, t, K=10):
    """Exactly the pipeline's windowing (no rng, as in assess line 414) + _disjointness."""
    win, Keff = dd._assign_windows(np.asarray(t, float), K, False)
    return dd._disjointness(np.asarray(X, float), win, Keff), win, Keff


def D_shuffle_of(X, t, K=10):
    """Exactly assess lines 424-428: permute t with default_rng(0), re-window, _disjointness.
    NOTE: X is NOT permuted (matches the code)."""
    t = np.asarray(t, float)
    win, Keff = dd._assign_windows(t, K, False)
    sh = np.random.default_rng(0).permutation(len(t))
    wsh, Ksh = dd._assign_windows(t[sh], Keff, False)
    return dd._disjointness(np.asarray(X, float), wsh, Ksh)


# --------------------------------------------------------------------- D1 duplicates
print("== D1: duplicates on stable synth (stationary X, sorted t) ==", flush=True)
Xs, ys, ts, _ = dd._synth("stable")            # n=12000 d=10, t sorted, X iid N(0,1)
d1 = {}
D0, _, _ = D_of(Xs, ts)
d1["m=1"] = D0
print(f"  m=1  (no dup)  D = {D0:.4f}", flush=True)
for m in (2, 5, 20):
    Xd = np.repeat(Xs, m, axis=0)
    td = np.repeat(ts, m)                      # SAME t for each duplicate
    D, _, _ = D_of(Xd, td)
    d1[f"m={m}"] = D
    print(f"  m={m:<3d}          D = {D:.4f}", flush=True)
OUT["D1_duplicates_stable"] = d1

# --------------------------------------------------------------------- D2 entities
print("== D2: entity recurrence, E=300, 40 obs each, obs-noise 0.1, NO drift ==", flush=True)
rng = np.random.default_rng(0)
E, R, DIM = 300, 40, 10
lat = rng.normal(0, 1, (E, DIM))
ent = np.repeat(np.arange(E), R)

# D2: random times -> each entity spans all windows
t2 = rng.random(E * R)
X2 = lat[ent] + 0.1 * rng.normal(0, 1, (E * R, DIM))
o = np.argsort(t2, kind="stable")
X2o, t2o = X2[o], t2[o]
D2, _, _ = D_of(X2o, t2o)
print(f"  D2  (random times, entity spans all windows)  D = {D2:.4f}", flush=True)
OUT["D2_entities_random_times"] = D2

# D2b: cohorts -> each entity lives in ONE narrow band (width 0.01)
centers = rng.uniform(0, 0.99, E)
t2b = np.repeat(centers, R) + rng.uniform(0, 0.01, E * R)
X2b = lat[ent] + 0.1 * rng.normal(0, 1, (E * R, DIM))
o = np.argsort(t2b, kind="stable")
X2bo, t2bo, entbo = X2b[o], t2b[o], ent[o]
D2b, win2b, K2b = D_of(X2bo, t2bo)
print(f"  D2b (cohorts: entity in one band, iid latents = NO covariate shift)  D = {D2b:.4f}",
      flush=True)
OUT["D2b_entity_cohorts_no_drift"] = D2b

# population-level truth for D2b: one obs per entity, fresh entities -> what would an honest
# distribution-level classifier get? (latents iid across windows -> 0.5)
lat_a = rng.normal(0, 1, (3000, DIM))
lat_b = rng.normal(0, 1, (3000, DIM))
Xpop = np.vstack([lat_a, lat_b])
tpop = np.r_[np.zeros(3000), np.ones(3000)] + rng.uniform(0, 0.01, 6000)
o = np.argsort(tpop, kind="stable")
Dpop, _, _ = D_of(Xpop[o], tpop[o], K=2)
print(f"  reference: fresh iid entities, 1 obs each, 2 windows  D = {Dpop:.4f}", flush=True)
OUT["D2b_reference_fresh_entities"] = Dpop

# --------------------------------------------------------------------- D3 mild drift
print("== D3: + mild mean drift 0.5*t on 2 features ==", flush=True)
d3 = {}
# (a) cohorts + drift
X3a = X2bo.copy()
X3a[:, :2] += 0.5 * t2bo[:, None]
D3a, _, _ = D_of(X3a, t2bo)
d3["cohorts_plus_drift"] = D3a
print(f"  (a) cohorts + drift          D = {D3a:.4f}", flush=True)
# (b) random-time entities + same drift
X3b = X2o.copy()
X3b[:, :2] += 0.5 * t2o[:, None]
D3b, _, _ = D_of(X3b, t2o)
d3["random_entities_plus_drift"] = D3b
print(f"  (b) random-time entities + drift  D = {D3b:.4f}", flush=True)
# (c) pure iid rows + same drift (no entity structure at all)
t3c = np.sort(rng.random(E * R))
X3c = rng.normal(0, 1, (E * R, DIM))
X3c[:, :2] += 0.5 * t3c[:, None]
D3c, _, _ = D_of(X3c, t3c)
d3["iid_rows_plus_drift"] = D3c
print(f"  (c) iid rows + drift (true geometry)  D = {D3c:.4f}", flush=True)
OUT["D3_mild_drift"] = d3

# --------------------------------------------------------------------- D4 shuffle control
print("== D4: time-shuffle control mechanics ==", flush=True)
d4 = {}

# coarse timestamps: quantize t to 45 unique values (ecom-like n_unique_t=45)
def quant(t, U=45):
    return np.floor(np.asarray(t) * U) / U

# D2b cohorts, coarse t
t2b_q = quant(t2bo)
Dq, _, _ = D_of(X2bo, t2b_q)
Dq_sh = D_shuffle_of(X2bo, t2b_q)
d4["cohorts_coarse"] = {"D": Dq, "D_shuffle": Dq_sh}
print(f"  cohorts, coarse t(U=45):   D = {Dq:.4f}   D_shuffle = {Dq_sh:.4f}", flush=True)

# D3a cohorts+drift, coarse t
Dq3, _, _ = D_of(X3a, t2b_q)
Dq3_sh = D_shuffle_of(X3a, t2b_q)
d4["cohorts_drift_coarse"] = {"D": Dq3, "D_shuffle": Dq3_sh}
print(f"  cohorts+drift, coarse t:   D = {Dq3:.4f}   D_shuffle = {Dq3_sh:.4f}", flush=True)

# duplicates under shuffle (m=5): do duplicated rows keep D_shuffle high?
Xd5 = np.repeat(Xs, 5, axis=0)
td5 = np.repeat(ts, 5)
Dd5_sh = D_shuffle_of(Xd5, td5)
d4["duplicates_m5"] = {"D": d1["m=5"], "D_shuffle": Dd5_sh}
print(f"  duplicates m=5, stable:    D = {d1['m=5']:.4f}   D_shuffle = {Dd5_sh:.4f}", flush=True)

# skewed window sizes + drift: does the old[:nn]/fut[:nn] head-truncation of a BIG shuffled
# window (np.where returns time-ordered row indices) create a spurious early-vs-uniform
# contrast that pushes D_shuffle > 0.6 on drifting data?
n4 = 45000
U = 45
counts = (np.exp(rng.uniform(0, 3.0, U)))          # heavily unequal rows-per-timestamp
counts = (counts / counts.sum() * n4).astype(int)
counts[-1] += n4 - counts.sum()
vals = np.arange(U) / U
t4 = np.repeat(vals, counts)                       # sorted by construction (time-ordered rows)
X4 = rng.normal(0, 1, (n4, DIM))
X4[:, :3] += 2.0 * t4[:, None]                     # strong covariate drift
D4v, win4, K4 = D_of(X4, t4)
D4v_sh = D_shuffle_of(X4, t4)
sizes = [int((win4 == k).sum()) for k in range(K4)]
d4["skewed_windows_drift"] = {"D": D4v, "D_shuffle": D4v_sh, "window_sizes": sizes}
print(f"  skewed windows + drift:    D = {D4v:.4f}   D_shuffle = {D4v_sh:.4f}", flush=True)
print(f"    window sizes: {sizes}", flush=True)

# diagnose mechanism: mean ORIGINAL time of the rows _disjointness actually uses after shuffle
t4s = np.asarray(t4, float)
sh = np.random.default_rng(0).permutation(len(t4s))
wsh, Ksh = dd._assign_windows(t4s[sh], K4, False)
by_w = [np.where(wsh == k)[0] for k in range(Ksh)]
old = by_w[0]
diag = []
for j in range(max(1, Ksh // 2), Ksh):
    fut = by_w[j]
    if len(fut) < 40 or len(old) < 40:
        continue
    nn = min(len(old), len(fut), 5000)
    diag.append({"j": j, "nn": nn, "len_old": len(old), "len_fut": len(fut),
                 "mean_t_old_used": float(t4s[old[:nn]].mean()),
                 "mean_t_fut_used": float(t4s[fut[:nn]].mean()),
                 "mean_t_fut_all": float(t4s[fut].mean())})
d4["skewed_shuffle_diagnostic"] = diag
for row in diag:
    print(f"    j={row['j']}: nn={row['nn']} |old|={row['len_old']} |fut|={row['len_fut']} "
          f"mean_t(old used)={row['mean_t_old_used']:.3f} "
          f"mean_t(fut used)={row['mean_t_fut_used']:.3f} "
          f"mean_t(fut all)={row['mean_t_fut_all']:.3f}", flush=True)

OUT["D4_shuffle"] = d4

out_path = (r"C:\Users\joon\AppData\Local\Temp\claude"
            r"\C--Users-joon-Desktop-ExplainableTab"
            r"\e9542da3-1a26-4cb3-9fd3-ce63bc60e8d9\scratchpad\exp-dgate\d1_d4_results.json")
with open(out_path, "w") as f:
    json.dump(OUT, f, indent=2, default=float)
print(f"saved {out_path}", flush=True)
