"""Run the 5 synth controls for ONE model class; write <class>_results.json.
Usage: python run_one_class.py {hgb|linear|rf|knn}
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

mc = sys.argv[1]

DD_PATH = r"C:\Users\joon\Desktop\ExplainableTab\scripts\run_deployment_decay.py"
spec = importlib.util.spec_from_file_location("dd", DD_PATH)
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)

if mc != "hgb":
    C, R = make_shims(mc)
    dd.HistGradientBoostingClassifier = C
    dd.HistGradientBoostingRegressor = R

CONTROLS = ["concept", "covariate", "stable", "covariate_mc", "nuisance_proxy"]
OUT = HERE / f"{mc}_results.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {}

for kind in CONTROLS:
    if kind in res and "error" not in res[kind]:
        print(f"[{mc}] skip {kind}", flush=True)
        continue
    dd._FIT_WARNED.clear()
    X, y, t, task = dd._synth(kind)
    t0 = time.time()
    try:
        r = dd.assess(f"synth_{kind}", X, y, t, task, K=10, n_seeds=5, max_train=6000)
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
    res[kind] = cell
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"[{mc}] {kind} => {cell.get('verdict', 'ERROR')} "
          f"stale={cell.get('staleness_harm')} D_strip={cell.get('D_strip')} "
          f"({cell['wall_s']}s)", flush=True)
print(f"[{mc}] ALL DONE", flush=True)
