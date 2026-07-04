"""R6: sberbank-like regression control. NO concept drift (fixed rule), but:
   - n=19000, d=392 features (sberbank_housing has 392 feats, n=18847)
   - 30% MCAR NaN
   - several drifting covariates (linear ramps of varying strength), some predictive
   - fixed NONLINEAR rule on a mix of drifting + stationary features
   Question: does D_strip hit ~1.0 and does staleness_harm drift above the 0.02 floor
   with CI>0 (i.e. a false DEPLOYMENT-CONCEPT) under a fixed rule?
"""
import sys, json, importlib.util
from pathlib import Path

REPO = Path(r"C:\Users\joon\Desktop\ExplainableTab")
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location(
    "dd", str(REPO / "scripts" / "run_deployment_decay.py"))
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)

import numpy as np

OUT = Path(__file__).parent / "r6_results.json"


def synth_sberbank_like(seed=0, n=19000, d=392, nan_frac=0.30):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n))
    X = rng.normal(0, 1, (n, d))
    # drifting covariates: 6 features ramp with t at varying strengths; 3 of them predictive
    drift_feats = [0, 1, 2, 3, 4, 5]
    strengths = [6.0, 4.0, 3.0, 2.0, 5.0, 1.5]
    for j, s in zip(drift_feats, strengths):
        X[:, j] = X[:, j] + s * t
    # fixed NONLINEAR rule: mixes drifting (0,1,2) and stationary (6..10) features
    y = (np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 6]
         + 0.5 * np.tanh(X[:, 1]) + 0.4 * X[:, 7] * (X[:, 8] > 0)
         + 0.3 * np.abs(X[:, 2]) + rng.normal(0, 0.3, n))
    # 30% MCAR NaN on features (not target, not time)
    mask = rng.random((n, d)) < nan_frac
    X[mask] = np.nan
    return X, y, t


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    X, y, t = synth_sberbank_like()
    r = dd.assess("reg_sberbank_like_fixedrule", X, y, t, "regression",
                  K=10, n_seeds=n_seeds, max_train=6000)
    print(json.dumps({k: r[k] for k in (
        "dataset", "verdict", "staleness_harm", "staleness_harm_ci", "recency_gain",
        "recency_gain_ci", "decay", "D_strip", "D_full", "D_shuffle", "injected_staleness",
        "delta_staleness", "n_proxy_stripped", "trust")}, default=float), flush=True)
    prev = json.loads(OUT.read_text())["rows"] if OUT.exists() else []
    OUT.write_text(json.dumps({"rows": prev + [r]}, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
