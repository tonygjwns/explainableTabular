"""V4-B probe — does the validated toolkit find REAL, HARMFUL concept drift in ADVERSARIAL
domains (fraud / malware / spam), where attackers actively change the rule P(y|x)?

Seven dissolutions established that industrial tabular temporal shift is covariate-dominated,
concept is mostly absent/unmeasurable, and even the near-disjoint support is BENIGN (V4.0). The
toolkit is domain-invariant (proven on ACS). Theory says concept lives only where P(y|x) actually
moves — and adversarial domains are the one place it is *designed* to (an attacker rotates the
decision rule to evade). This probe takes the toolkit, intact, to that frontier.

Discipline (lessons baked in so this can't dissolve like the others): a single dataset counts as
a REAL HARMFUL CONCEPT positive ONLY if it passes ALL of:
  (1) measurable: within-overlap gap is computable (ess ≥ 5%, common support survives),
  (2) real:       gap − permutation-placebo > the 0.041 noise floor (not home-field),
  (3) not autocorrelation: a lagged-label ablation does NOT collapse the gap (the Elec2 trap),
  (4) harmful:    the shift actually degrades prediction (early→late perf drop OR conformal
                  under-coverage) — concept that doesn't hurt is not worth a method.
PRE-REGISTERED: if >=1 adversarial dataset is REAL-HARMFUL-CONCEPT -> the toolkit found concept
where it lives; the positive direction is "measurable, exploitable concept drift lives in
adversarial domains, and here is the validated protocol + benchmark." If 0/N -> concept is absent
even adversarially -> a STRONGER general negative (the measurement paper, now domain-broad).

Model-light (sklearn). Generic loader: point it at any time-ordered tabular CSV/parquet.

    python scripts/run_adversarial_probe.py --synth-adversarial            # positive-control smoke
    python scripts/run_adversarial_probe.py --csv baf.parquet --target fraud_bool --time month
    python scripts/run_adversarial_probe.py --csv ieee.csv --target isFraud --time TransactionDT \
        --drop TransactionID
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error  # noqa: E402

from src.analysis.drift_measure import (  # noqa: E402
    covariate_shift_auc, concept_within_overlap,
)

FLOOR = 0.041


def _perf_drop(Xe, ye, Xl, yl, task, seed=0):
    Xe = np.nan_to_num(np.asarray(Xe, float)); Xl = np.nan_to_num(np.asarray(Xl, float))
    Xtr, Xho, ytr, yho = train_test_split(Xe, ye, test_size=0.4, random_state=seed)
    if task == "regression":
        m = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(Xtr, ytr)
        r = lambda X, y: float(np.sqrt(mean_squared_error(y, m.predict(X))))
        return r(Xl, yl) - r(Xho, yho)
    if len(np.unique(ytr)) < 2:
        return None
    m = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(Xtr, ytr)
    if task == "binclass":
        sc = lambda X, y: roc_auc_score(y, m.predict_proba(X)[:, 1]) if len(np.unique(y)) > 1 else np.nan
    else:
        sc = lambda X, y: accuracy_score(y, m.predict(X))
    a, b = sc(Xho, yho), sc(Xl, yl)
    return float(a - b) if np.isfinite(a) and np.isfinite(b) else None


def _conformal_under(Xe, ye, Xl, yl, task, seed=0, alpha=0.10):
    Xe = np.nan_to_num(np.asarray(Xe, float)); Xl = np.nan_to_num(np.asarray(Xl, float))
    Xtr, Xcal, ytr, ycal = train_test_split(Xe, ye, test_size=0.4, random_state=seed)
    if task == "regression":
        m = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(Xtr, ytr)
        q = np.quantile(np.abs(ycal - m.predict(Xcal)), 1 - alpha)
        cov = float(np.mean(np.abs(yl - m.predict(Xl)) <= q))
        return (1 - alpha) - cov
    if len(np.unique(ytr)) < 2:
        return None
    m = HistGradientBoostingClassifier(max_iter=300, random_state=seed).fit(Xtr, ytr)
    cls = list(m.classes_); idx = {c: i for i, c in enumerate(cls)}
    s = 1.0 - m.predict_proba(Xcal)[np.arange(len(ycal)), [idx.get(c, 0) for c in ycal]]
    q = np.quantile(s, 1 - alpha)
    inset = m.predict_proba(Xl) >= (1 - q)
    cov = float(np.mean([inset[i, idx[c]] if c in idx else False for i, c in enumerate(yl)]))
    return (1 - alpha) - cov


def _gap(Xe, ye, Xl, yl, task, seed, permute=False):
    r = concept_within_overlap(Xe, ye, Xl, yl, task, seed=seed, permute_time=permute)
    return (r.get("concept_gap_within_overlap") if r.get("measurable") else None), r.get("ess_pct")


def assess(name, Xe, ye, Xl, yl, t_raw_e, task, seed=0):
    cov = covariate_shift_auc(Xe, Xl, seed=seed).get("auc")
    gap, ess = _gap(Xe, ye, Xl, yl, task, seed, False)
    plac, _ = _gap(Xe, ye, Xl, yl, task, seed, True)
    # autocorrelation guard (binclass): add lagged label as a feature; if gap collapses, it was serial
    gap_lag = None
    if task == "binclass" and t_raw_e is not None:
        oe = np.argsort(t_raw_e, kind="stable")
        ylag_e = np.concatenate([[ye[oe[0]]], ye[oe[:-1]]]).astype(float)
        # align lag back to original order
        lag_e = np.empty(len(ye)); lag_e[oe] = ylag_e
        Xe2 = np.hstack([Xe, lag_e[:, None]]); Xl2 = np.hstack([Xl, np.zeros((len(yl), 1))])
        gap_lag, _ = _gap(Xe2, ye, Xl2, yl, task, seed, False)
    drop = _perf_drop(Xe, ye, Xl, yl, task, seed)
    under = _conformal_under(Xe, ye, Xl, yl, task, seed)
    measurable = gap is not None and (ess is not None and ess >= 5.0)
    # real concept requires the gap ITSELF above floor (not just bias-corrected — a structurally
    # negative placebo can inflate gap-placebo while the raw gap is ~0/negative, as on BAF).
    real = bool(measurable and gap is not None and gap > FLOOR
                and plac is not None and (gap - plac) > FLOOR)
    not_autocorr = bool(gap_lag is None or task != "binclass" or gap_lag > FLOOR)
    harmful = bool((drop is not None and drop > (0.5 if task == "regression" else 0.05))
                   or (under is not None and under > 0.05))
    verdict = ("REAL-HARMFUL-CONCEPT" if (real and not_autocorr and harmful)
               else "autocorrelation" if (real and not not_autocorr)
               else "concept-but-benign" if (real and not harmful)
               else "concept~0" if measurable else "unmeasurable")
    return {"dataset": name, "task": task, "n_early": len(ye), "n_late": len(yl),
            "cov_auc": cov, "ess_pct": ess, "gap": gap, "placebo": plac,
            "gap_minus_placebo": (None if (gap is None or plac is None) else gap - plac),
            "gap_with_lag": gap_lag, "perf_drop": drop, "conformal_under": under,
            "verdict": verdict}


def _load_csv(path, target, time, drop, max_n):
    import pandas as pd
    from pandas.api import types as pt
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    if max_n and len(df) > max_n:
        df = df.sort_values(time).iloc[:: max(1, len(df) // max_n)].reset_index(drop=True)

    def to_num(col):                                  # robust to pandas extension dtypes
        if pt.is_numeric_dtype(col):
            return pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64")
        if pt.is_datetime64_any_dtype(col):
            return col.view("int64").to_numpy().astype("float64")
        return col.astype("category").cat.codes.to_numpy().astype("float64")

    y_raw = df[target]
    tnum = to_num(df[time])
    drop_cols = set([target, time] + list(drop or []))
    feats = [c for c in df.columns if c not in drop_cols]
    X = np.column_stack([to_num(df[c]) for c in feats])
    yv = y_raw.to_numpy()
    nuniq = len(pd.unique(y_raw.dropna()))
    task = ("binclass" if nuniq == 2 else
            "regression" if (pt.is_float_dtype(y_raw) and nuniq > 20) else "multiclass")
    if task != "regression":
        cls = {c: i for i, c in enumerate(sorted(pd.unique(y_raw.dropna())))}
        y = np.array([cls.get(v, 0) for v in yv])
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype="float64")
    o = np.argsort(tnum, kind="stable"); med = len(o) // 2
    em, lm = o[:med], o[med:]
    return X[em], y[em], X[lm], y[lm], tnum[em], task, len(feats)


def _synth_adversarial(seed=0, n=8000, d=10):
    """Attacker rotates the decision rule over time (P(y|x) moves), modest covariate shift.
    Should read REAL-HARMFUL-CONCEPT: measurable, survives placebo, not autocorrelation, harmful."""
    rng = np.random.default_rng(seed); t = np.sort(rng.random(n))
    X = rng.normal(0, 1, (n, d))
    ang = 2.5 * t                                      # rule rotates smoothly with time
    score = np.cos(ang) * X[:, 0] + np.sin(ang) * X[:, 1]
    y = (3 * score + rng.normal(0, .4, n) > 0).astype(int)
    tr = t < 0.6
    return X[tr], y[tr], X[~tr], y[~tr], np.arange(tr.sum()), "binclass", d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="path to a time-ordered tabular CSV/parquet")
    ap.add_argument("--target", default=None); ap.add_argument("--time", default=None)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--name", default=None)
    ap.add_argument("--max-n", type=int, default=40000)
    ap.add_argument("--synth-adversarial", action="store_true")
    args = ap.parse_args()
    out_dir = Path("results/phase1/adversarial_probe"); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def show(r):
        f = lambda x: f"{x:+.3f}" if isinstance(x, (int, float)) else "   -"
        print(f"  {r['dataset'][:22]:22s} cov={f(r['cov_auc'])} ess={f(r['ess_pct'])} "
              f"gap={f(r['gap'])} plc={f(r['placebo'])} g-p={f(r['gap_minus_placebo'])} "
              f"lag={f(r['gap_with_lag'])} drop={f(r['perf_drop'])} cfU={f(r['conformal_under'])} "
              f"=> {r['verdict']}")

    if args.synth_adversarial:
        print("\n==== SYNTH adversarial control (rotating rule => expect REAL-HARMFUL-CONCEPT) ====")
        Xe, ye, Xl, yl, tre, task, d = _synth_adversarial()
        r = assess("synth_adversarial", Xe, ye, Xl, yl, tre, task); rows.append(r); show(r)
        # negative control: covariate-only (fixed rule) => expect concept~0/benign
        rng = np.random.default_rng(1); n = 8000; t = np.sort(rng.random(n)); X = rng.normal(0, 1, (n, 6))
        X[:, 1] += 5 * t; y = (3 * X[:, 0] + rng.normal(0, .4, n) > 0).astype(int); tr = t < 0.6
        r2 = assess("synth_covariate_only", X[tr], y[tr], X[~tr], y[~tr], np.arange(tr.sum()), "binclass")
        rows.append(r2); show(r2)
        (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2, default=float))
        print("\n  EXPECT: adversarial => REAL-HARMFUL-CONCEPT ; covariate_only => concept~0/benign")
        print(f"  wrote {out_dir}/summary.json"); return

    if not (args.csv and args.target and args.time):
        print("provide --csv --target --time  (or --synth-adversarial)"); return
    Xe, ye, Xl, yl, tre, task, nf = _load_csv(args.csv, args.target, args.time, args.drop, args.max_n)
    name = args.name or Path(args.csv).stem
    print(f"\n==== ADVERSARIAL probe [{name}] task={task} feats={nf} n={len(ye)}/{len(yl)} ====")
    r = assess(name, Xe, ye, Xl, yl, tre, task); rows.append(r); show(r)
    print("\n  PRE-REG: 'REAL-HARMFUL-CONCEPT' here => the toolkit found exploitable concept where it")
    print("  lives (adversarial). 'concept~0/unmeasurable/autocorrelation/benign' => concept absent even")
    print("  adversarially => the broad measurement negative. Run several adversarial datasets to decide.")
    blob = json.loads((out_dir / "summary.json").read_text()) if (out_dir / "summary.json").exists() else {"rows": []}
    blob.setdefault("rows", []).append(r)
    (out_dir / "summary.json").write_text(json.dumps(blob, indent=2, default=float))
    print(f"\n  appended to {out_dir}/summary.json  <-- send me this")


if __name__ == "__main__":
    main()
