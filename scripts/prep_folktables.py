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


def main():
    if len(sys.argv) < 3:
        print("usage: python scripts/prep_folktables.py <STATE> <out.parquet> [task=income]")
        return
    state, out = sys.argv[1], sys.argv[2]
    from folktables import ACSDataSource, ACSIncome
    frames = []
    for year in YEARS:
        src = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person",
                            root_dir=str(Path("data") / "folktables"))
        raw = src.get_data(states=[state], download=True)
        feats, label, _ = ACSIncome.df_to_pandas(raw)
        df = feats.copy()
        df["label"] = label.astype(int).to_numpy().ravel()
        df["YEAR"] = year
        frames.append(df)
        print(f"  {state} {year}: n={len(df)}, pos_rate={df['label'].mean():.3f}", flush=True)
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(out)
    print(f"wrote {out}: n={len(full)}, feats={full.shape[1] - 2}, years={sorted(set(full['YEAR']))}")


if __name__ == "__main__":
    main()
