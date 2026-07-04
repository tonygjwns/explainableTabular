"""Regression ground-truth controls for the deployment-decay instrument (sberbank cell audit).

Imports scripts/run_deployment_decay.py faithfully and calls dd.assess() on synthetic
regression datasets mirroring the repo's own _synth style (n=12000, d=10, t=sorted uniform).

R1 reg-stable            : y = 3*X0 + eps, stationary X.           Expect stale~0.
R2 reg-concept           : y = cos(3t)*X0 + sin(3t)*X1 + eps.      Expect DEPLOYMENT-CONCEPT.
R3 reg-covariate-linear  : X0 += 6t, y = 3*X0 + 0.8*X1 + eps.      FIXED linear rule, drifting X.
R4 reg-covariate-nonlin  : X0 += 6t, y = sin(1.5*X0)+0.8*X1+eps.   FIXED nonlinear rule (KEY RUN;
                           regression twin of the repo's binclass 'covariate' control, line 604-607).
R5 reg-hetero-noise      : y = 3*X0 + eps(t), noise std 0.3->1.5.  P(y|x) VARIANCE changes only.
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

OUT = Path(__file__).parent / "reg_controls_results.json"


def synth_reg(kind, seed=0, n=12000, d=10):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n))
    X = rng.normal(0, 1, (n, d))
    if kind == "reg_stable":
        y = 3 * X[:, 0] + rng.normal(0, 0.3, n)
    elif kind == "reg_concept":
        ang = 3.0 * t
        y = np.cos(ang) * X[:, 0] + np.sin(ang) * X[:, 1] + rng.normal(0, 0.3, n)
    elif kind == "reg_cov_linear":
        X[:, 0] = X[:, 0] + 6.0 * t
        y = 3 * X[:, 0] + 0.8 * X[:, 1] + rng.normal(0, 0.3, n)
    elif kind == "reg_cov_nonlinear":
        X[:, 0] = X[:, 0] + 6.0 * t
        y = np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1] + rng.normal(0, 0.3, n)
    elif kind == "reg_hetero_noise":
        sd = 0.3 + 1.2 * t                      # noise std grows 0.3 -> 1.5 with t
        y = 3 * X[:, 0] + rng.normal(0, 1, n) * sd
    else:
        raise ValueError(kind)
    return X, y, t


def main():
    kinds = sys.argv[1:] if len(sys.argv) > 1 else [
        "reg_stable", "reg_concept", "reg_cov_linear", "reg_cov_nonlinear", "reg_hetero_noise"]
    n_seeds = int(next((a.split("=")[1] for a in kinds if a.startswith("seeds=")), 5))
    kinds = [k for k in kinds if not k.startswith("seeds=")]
    rows = []
    for kind in kinds:
        X, y, t = synth_reg(kind)
        r = dd.assess(kind, X, y, t, "regression", K=10, n_seeds=n_seeds, max_train=6000)
        rows.append(r)
        print(json.dumps({k: r[k] for k in (
            "dataset", "verdict", "staleness_harm", "staleness_harm_ci", "recency_gain",
            "recency_gain_ci", "decay", "D_strip", "D_full", "D_shuffle", "injected_staleness",
            "delta_staleness", "n_proxy_stripped", "trust")}, default=float), flush=True)
    prev = json.loads(OUT.read_text())["rows"] if OUT.exists() else []
    OUT.write_text(json.dumps({"rows": prev + rows}, indent=2, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
