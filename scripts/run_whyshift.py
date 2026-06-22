"""V3.2 C5 — earn the spatial-vs-temporal contrast on WhyShift/folktables. (PLAN_V3 §C5)

WhyShift [Liu et al. 2023] finds Y|X-shift dominant on SPATIAL tabular settings and notes
X-shift dominant in its one TEMPORAL setting (ACS Time). We run OUR toolkit (cov_AUC,
overlap, within-overlap concept gap + permutation placebo) on the same data family
(ACS via folktables) to test the contrast directly with OUR frame:

  PREDICTION: spatial pairs (state A -> state B, same year) => concept MEASURABLE and
  gap > 0 (Y|X-dominant); temporal pairs (same state, year Y0 -> Y1) => higher cov_AUC /
  smaller gap (X-dominant). If borne out, the "spatial=Y|X / temporal=X" contrast is
  earned with our instrument (and the toolkit generalizes beyond TabReD).

  NOTE: folktables features (~10 raw demographic) are far less time-leaked than TabReD's
  261 engineered features, so the representation-dependence (§6) should bite LESS here --
  a useful cross-check that "unmeasurability" tracks feature engineering, not the toolkit.

Needs `folktables` (pip install folktables) + network for the ACS download (server).
Skips gracefully if unavailable.

    python scripts/run_whyshift.py
    python scripts/run_whyshift.py --states CA TX NY FL PA --years 2014 2018 --task income
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.analysis.drift_measure import (  # noqa: E402
    covariate_shift_auc, concept_within_overlap,
)

MAX_N = 20000


def _load_acs(states, years, task):
    from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage, ACSMobility
    prob = {"income": ACSIncome, "pubcov": ACSPublicCoverage, "mobility": ACSMobility}[task]
    data = {}   # (state, year) -> (X, y)
    for yr in years:
        src = ACSDataSource(survey_year=str(yr), horizon="1-Year", survey="person")
        for st in states:
            df = src.get_data(states=[st], download=True)
            X, y, _ = prob.df_to_numpy(df)
            data[(st, yr)] = (np.asarray(X, float), np.asarray(y).astype(int))
    return data


def _sub(X, y, n, seed):
    rng = np.random.default_rng(seed)
    if len(y) > n:
        ii = rng.choice(len(y), n, replace=False); return X[ii], y[ii]
    return X, y


def _measure(Xs, ys, Xt, yt, seed):
    Xs, ys = _sub(Xs, ys, MAX_N, seed); Xt, yt = _sub(Xt, yt, MAX_N, seed)
    cov = covariate_shift_auc(Xs, Xt, seed=seed).get("auc")
    con = concept_within_overlap(Xs, ys, Xt, yt, "binclass", seed=seed)
    plc = concept_within_overlap(Xs, ys, Xt, yt, "binclass", seed=seed, permute_time=True)
    meas = bool(con.get("measurable"))
    return {"cov_auc": cov, "measurable": meas,
            "gap": con.get("concept_gap_within_overlap") if meas else None,
            "placebo": plc.get("concept_gap_within_overlap") if plc.get("measurable") else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", default=["CA", "TX", "NY", "FL", "PA"])
    ap.add_argument("--years", nargs="+", default=["2014", "2018"])
    ap.add_argument("--task", default="income", choices=["income", "pubcov", "mobility"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path("results/phase1/whyshift"); out_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = _load_acs(args.states, args.years, args.task)
    except Exception as e:  # folktables missing / no network
        print(f"SKIP (folktables unavailable): {type(e).__name__}: {e}")
        print("  install: pip install folktables  (needs network for the ACS download)")
        return

    y_late = args.years[-1]
    spatial, temporal = [], []
    print(f"\n==== WhyShift contrast (task={args.task}) ====")
    print(f"  {'setting':28s}{'cov_AUC':>8s}{'meas':>6s}{'gap':>9s}{'placebo':>9s}")

    def show(tag, src, tgt):
        Xs, ys = data[src]; Xt, yt = data[tgt]
        r = _measure(Xs, ys, Xt, yt, args.seed); r["setting"] = tag; r["src"] = src; r["tgt"] = tgt
        gt = f"{r['gap']:+.3f}" if isinstance(r["gap"], (int, float)) else "   -"
        pt = f"{r['placebo']:+.3f}" if isinstance(r["placebo"], (int, float)) else "   -"
        print(f"  {tag:28s}{r['cov_auc']:8.3f}{str(r['measurable']):>6s}{gt:>9s}{pt:>9s}")
        return r

    # SPATIAL: state A -> state B, same (latest) year
    print("  -- spatial (state->state, same year) --")
    base = args.states[0]
    for st in args.states[1:]:
        r = show(f"{base}->{st} @{y_late}", (base, y_late), (st, y_late))
        spatial.append(r)
    # TEMPORAL: same state, year0 -> year1
    if len(args.years) >= 2:
        print("  -- temporal (same state, year0->year1) --")
        for st in args.states:
            r = show(f"{st} {args.years[0]}->{y_late}", (st, args.years[0]), (st, y_late))
            temporal.append(r)

    def agg(rs):
        gaps = [r["gap"] for r in rs if isinstance(r["gap"], (int, float))]
        covs = [r["cov_auc"] for r in rs]
        meas = [r["measurable"] for r in rs]
        return (float(np.mean(gaps)) if gaps else float("nan"),
                float(np.mean(covs)), float(np.mean(meas)))
    sg, sc, sm = agg(spatial); tg, tc, tm = agg(temporal)
    print("\n  ==== aggregate ====")
    print(f"  SPATIAL : mean cov_AUC={sc:.3f}  measurable={100*sm:.0f}%  mean concept_gap={sg:+.3f}")
    print(f"  TEMPORAL: mean cov_AUC={tc:.3f}  measurable={100*tm:.0f}%  mean concept_gap={tg:+.3f}")
    print("  PREDICTION: spatial gap > temporal gap, temporal cov_AUC > spatial (X-dominant).")
    print("  If borne out, the spatial=Y|X / temporal=X contrast is earned with our frame.")
    (out_dir / "summary.json").write_text(json.dumps(
        {"task": args.task, "spatial": spatial, "temporal": temporal,
         "agg": {"spatial": [sg, sc, sm], "temporal": [tg, tc, tm]}}, indent=2, default=float))
    print(f"\n  wrote {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
