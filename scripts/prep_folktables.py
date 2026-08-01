"""Materialize a folktables ACS task across YEARS into one parquet with a YEAR time column —
the WhyShift temporal bridge (PREREG_DEPLOYMENT_V2 §5 Phase 4): WhyShift (NeurIPS D&B 2023)
mapped Y|X- vs X-shift on ACS tasks across *states*; we run the deployment-decay instrument on
the same task across *years*, so the two maps meet on shared data.

Years 2014-2018 only, deliberately: (a) ACS renamed RELP->RELSHIPP in 2019, which breaks the
canonical ACSIncome feature list on 2019+ 1-Year files; (b) 2020's standard 1-Year release was
replaced by experimental weights (COVID nonresponse) and would leave a gap anyway. Five yearly
windows is coarse but honest — run the instrument with --by-value so windows = calendar years.

    python scripts/prep_folktables.py CA acs_income_CA.parquet          # ~5 x 1-Year downloads
    python scripts/run_deployment_decay.py --csv acs_income_CA.parquet \
        --target label --time YEAR --by-value --n-seeds 10 --name acs_income_CA

Requires `pip install folktables` (downloads census data into ./data/folktables on first use).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

YEARS = (2014, 2015, 2016, 2017, 2018)


TASKS = {"income": "ACSIncome", "pubcov": "ACSPublicCoverage",
         "employment": "ACSEmployment", "mobility": "ACSMobility"}


def main():
    if len(sys.argv) < 3:
        print(f"usage: python scripts/prep_folktables.py <STATE> <out.parquet> "
              f"[task={'|'.join(TASKS)}]  (default income)")
        return
    state, out = sys.argv[1], sys.argv[2]
    task = sys.argv[3] if len(sys.argv) > 3 else "income"
    if task not in TASKS:
        print(f"unknown task {task!r}; options: {list(TASKS)}"); return
    import folktables
    from folktables import ACSDataSource
    problem = getattr(folktables, TASKS[task])
    print(f"task={task} -> folktables.{TASKS[task]}  state={state}  years={list(YEARS)}", flush=True)
    frames = []
    for year in YEARS:
        src = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person",
                            root_dir=str(Path("data") / "folktables"))
        raw = src.get_data(states=[state], download=True)
        feats, label, _ = problem.df_to_pandas(raw)
        df = feats.copy()
        df["label"] = label.astype(int).to_numpy().ravel()
        df["YEAR"] = year
        frames.append(df)
        print(f"  {state} {year}: n={len(df)}, pos_rate={df['label'].mean():.3f}", flush=True)
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(out)
    print(f"wrote {out}: n={len(full)}, feats={full.shape[1] - 2}, years={sorted(set(full['YEAR']))}")
    # per-year positive rate is the first thing to eyeball on the pubcov task: a Medicaid-expansion
    # state should show a level shift at its implementation year, a non-expansion state should not.
    # This is descriptive only -- the verdict comes from the instrument, not from this ramp.
    print("  pos_rate by year: " + "  ".join(
        f"{y}:{full[full['YEAR'] == y]['label'].mean():.3f}" for y in sorted(set(full["YEAR"]))))


if __name__ == "__main__":
    main()
