"""D6: (a) honest-geometry calibration: iid rows, mean drift s*t on 2 features — what s makes
D cross 0.96/hit 1.0 when there is NO memorization channel?
(b) the cheap fix: group-aware split (split by entity/base-row, not by row) inside a
_disjointness clone — does it deflate D1 duplicates and D2b cohorts back toward truth?"""
import importlib.util
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

SPEC = importlib.util.spec_from_file_location(
    "dd", r"C:\Users\joon\Desktop\ExplainableTab\scripts\run_deployment_decay.py")
dd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dd)

OUT = {}
rng = np.random.default_rng(2)
n, DIM = 12000, 10

# (a) honest calibration sweep
cal = {}
for s in (0.5, 1.0, 2.0, 4.0, 6.0, 8.0):
    t = np.sort(rng.random(n))
    X = rng.normal(0, 1, (n, DIM))
    X[:, :2] += s * t[:, None]
    win, K = dd._assign_windows(t, 10, False)
    D = dd._disjointness(X, win, K)
    cal[f"s={s}"] = D
    print(f"  iid drift s={s}: D = {D:.4f}", flush=True)
OUT["honest_drift_calibration"] = cal


# (b) group-aware _disjointness clone (only change: GroupShuffleSplit instead of row split)
def disjointness_grouped(X, win, Keff, groups, seed=0):
    by_w = [np.where(win == k)[0] for k in range(Keff)]
    old = by_w[0]
    if len(old) < 40:
        return None
    aucs = []
    for j in range(max(1, Keff // 2), Keff):
        fut = by_w[j]
        if len(fut) < 40:
            continue
        nn = min(len(old), len(fut), 5000)
        sel = np.r_[old[:nn], fut[:nn]]
        Xd = np.nan_to_num(X[sel])
        yd = np.r_[np.zeros(nn), np.ones(nn)]
        g = groups[sel]
        try:
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                            random_state=seed).split(Xd, yd, g))
            if len(np.unique(yd[tr])) < 2 or len(np.unique(yd[te])) < 2:
                continue
            m = HistGradientBoostingClassifier(max_iter=100, early_stopping=False,
                                               random_state=seed).fit(Xd[tr], yd[tr])
            aucs.append(roc_auc_score(yd[te], m.predict_proba(Xd[te])[:, 1]))
        except Exception:
            continue
    return float(np.median(aucs)) if aucs else None


fix = {}
# D1 duplicates m=5, stable
Xs, ys, ts, _ = dd._synth("stable")
m = 5
Xd = np.repeat(Xs, m, axis=0)
td = np.repeat(ts, m)
gd = np.repeat(np.arange(len(ts)), m)          # group = base row
win, K = dd._assign_windows(td, 10, False)
fix["dup_m5_row_split"] = dd._disjointness(Xd, win, K)
fix["dup_m5_group_split"] = disjointness_grouped(Xd, win, K, gd)
print(f"  dup m=5: row-split D = {fix['dup_m5_row_split']:.4f}  "
      f"group-split D = {fix['dup_m5_group_split']:.4f}", flush=True)

# D2b cohorts
rng2 = np.random.default_rng(0)
E, R = 300, 40
lat = rng2.normal(0, 1, (E, DIM))
ent = np.repeat(np.arange(E), R)
centers = rng2.uniform(0, 0.99, E)
t2b = np.repeat(centers, R) + rng2.uniform(0, 0.01, E * R)
X2b = lat[ent] + 0.1 * rng2.normal(0, 1, (E * R, DIM))
o = np.argsort(t2b, kind="stable")
X2b, t2b, ent_o = X2b[o], t2b[o], ent[o]
win, K = dd._assign_windows(t2b, 10, False)
fix["cohort_row_split"] = dd._disjointness(X2b, win, K)
fix["cohort_group_split"] = disjointness_grouped(X2b, win, K, ent_o)
print(f"  cohorts: row-split D = {fix['cohort_row_split']:.4f}  "
      f"group-split D = {fix['cohort_group_split']:.4f}", flush=True)
OUT["group_split_fix"] = fix

out_path = (r"C:\Users\joon\AppData\Local\Temp\claude"
            r"\C--Users-joon-Desktop-ExplainableTab"
            r"\e9542da3-1a26-4cb3-9fd3-ce63bc60e8d9\scratchpad\exp-dgate\d6_results.json")
with open(out_path, "w") as f:
    json.dump(OUT, f, indent=2, default=float)
print(f"saved {out_path}", flush=True)
