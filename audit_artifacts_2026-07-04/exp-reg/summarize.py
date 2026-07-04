import json
from pathlib import Path

HERE = Path(__file__).parent
rows = []
for f in ("reg_controls_results.json", "r6_results.json"):
    p = HERE / f
    if p.exists():
        rows += json.loads(p.read_text())["rows"]

SBER = dict(stale=0.023900573591226625, lo=0.01775786068965817, hi=0.030043286492795078)
print(f"{'dataset':28s} {'verdict':26s} {'stale':>8s} {'CI_lo':>8s} {'CI_hi':>8s} "
      f"{'D_strip':>7s} {'rec':>7s} {'decay':>7s} {'inj':>7s}  trust")
fmt = lambda v: f"{v:+.4f}" if isinstance(v, (int, float)) else "   None"
for r in rows:
    ci = r.get("staleness_harm_ci") or [None, None]
    print(f"{r['dataset']:28s} {r['verdict']:26s} {fmt(r.get('staleness_harm')):>8s} "
          f"{fmt(ci[0]):>8s} {fmt(ci[1]):>8s} {fmt(r.get('D_strip')):>7s} "
          f"{fmt(r.get('recency_gain')):>7s} {fmt(r.get('decay')):>7s} "
          f"{fmt(r.get('injected_staleness')):>7s}  {r.get('trust')}")
print(f"\nsberbank_housing (artifact)  DEPLOYMENT-CONCEPT          "
      f"{SBER['stale']:+.4f} {SBER['lo']:+.4f} {SBER['hi']:+.4f}  D_strip=1.000")
for r in rows:
    st = r.get("staleness_harm"); ci = r.get("staleness_harm_ci") or [None, None]
    if r["verdict"] == "DEPLOYMENT-CONCEPT" and "concept" not in r["dataset"]:
        overlap = (ci[0] is not None and ci[0] <= SBER["hi"] and ci[1] >= SBER["lo"])
        print(f"FALSE-CONCEPT: {r['dataset']} stale={st:+.4f} "
              f"CI[{ci[0]:+.4f},{ci[1]:+.4f}]  CI-overlaps-sberbank={overlap}")
