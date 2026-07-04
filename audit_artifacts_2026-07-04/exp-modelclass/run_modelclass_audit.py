"""Model-class dependence audit of the deployment-decay instrument (dossier Part 9B).

For each model class (hgb reference, linear=LR/Ridge, rf=RandomForest, knn=k=25),
monkeypatch dd.HistGradientBoostingClassifier/Regressor and re-run the FIVE synthetic
ground-truth controls (concept, covariate, stable, covariate_mc, nuisance_proxy) with
dd.assess(n_seeds=5). Everything downstream — staleness/decay/recency scoring
(_fit_score), the disjointness gate D (_disjointness), and the injection control —
uses the patched class, because both call sites resolve the name from dd's module
globals at call time.

Output: verdict_matrix.json (incrementally updated per cell) with verdict,
staleness_harm (+CI), recency_gain (+CI), decay, D_strip, D_full, injected_staleness,
flags, and wall time per cell.
"""
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shims import make_shims  # noqa: E402

DD_PATH = r"C:\Users\joon\Desktop\ExplainableTab\scripts\run_deployment_decay.py"
spec = importlib.util.spec_from_file_location("dd", DD_PATH)
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)

ORIG_CLF = dd.HistGradientBoostingClassifier
ORIG_REG = dd.HistGradientBoostingRegressor

CONTROLS = ["concept", "covariate", "stable", "covariate_mc", "nuisance_proxy"]
MODEL_CLASSES = ["hgb", "linear", "rf", "knn"]
N_SEEDS = 5
K = 10

OUT = HERE / "verdict_matrix.json"
results = {}
if OUT.exists():
    results = json.loads(OUT.read_text())  # resume support


def set_model_class(kind):
    if kind == "hgb":
        dd.HistGradientBoostingClassifier = ORIG_CLF
        dd.HistGradientBoostingRegressor = ORIG_REG
    else:
        C, R = make_shims(kind)
        dd.HistGradientBoostingClassifier = C
        dd.HistGradientBoostingRegressor = R


def main():
    for mc in MODEL_CLASSES:
        set_model_class(mc)
        results.setdefault(mc, {})
        for kind in CONTROLS:
            if kind in results[mc] and "error" not in results[mc][kind]:
                print(f"[skip] {mc}/{kind} already done", flush=True)
                continue
            dd._FIT_WARNED.clear()
            X, y, t, task = dd._synth(kind)
            t0 = time.time()
            try:
                r = dd.assess(f"synth_{kind}", X, y, t, task,
                              K=K, n_seeds=N_SEEDS, max_train=6000)
                cell = {k: r.get(k) for k in
                        ("verdict", "task", "n", "n_windows", "n_seeds_ok",
                         "decay", "decay_ci", "recency_gain", "recency_gain_ci",
                         "staleness_harm", "staleness_harm_ci",
                         "D_strip", "D_full", "D_shuffle", "injected_staleness",
                         "delta_staleness", "n_proxy_stripped", "trust")}
            except Exception as e:
                traceback.print_exc()
                cell = {"error": f"{type(e).__name__}: {e}"}
            cell["wall_s"] = round(time.time() - t0, 1)
            results[mc][kind] = cell
            OUT.write_text(json.dumps(results, indent=2, default=float))
            sh = cell.get("staleness_harm")
            shs = f"{sh:+.4f}" if isinstance(sh, float) else "None"
            ci = cell.get("staleness_harm_ci") or [None, None]
            cis = (f"[{ci[0]:+.4f},{ci[1]:+.4f}]"
                   if ci and ci[0] is not None else "[-]")
            ds = cell.get("D_strip")
            dss = f"{ds:.3f}" if isinstance(ds, float) else "None"
            inj = cell.get("injected_staleness")
            injs = f" inj={inj:+.4f}" if isinstance(inj, float) else ""
            print(f"[{mc:6s}] {kind:14s} stale={shs}{cis} D_strip={dss}{injs} "
                  f"=> {cell.get('verdict', 'ERROR')}  ({cell['wall_s']}s)  "
                  f"trust={cell.get('trust')}", flush=True)
    print("\nDONE. wrote", OUT, flush=True)

    # verdict matrix table
    print("\n=== VERDICT MATRIX (rows=control, cols=model class) ===", flush=True)
    hdr = f"{'control':14s} " + " ".join(f"{m:>26s}" for m in MODEL_CLASSES)
    print(hdr, flush=True)
    for kind in CONTROLS:
        row = f"{kind:14s} "
        for mc in MODEL_CLASSES:
            v = results.get(mc, {}).get(kind, {}).get("verdict", "?")
            row += f" {v:>26s}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
