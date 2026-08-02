#!/usr/bin/env bash
# Day-3 queue: localise the blind spot the ACS run exposed, and close two loose ends.
#
# The deployment probe read PA (documented Medicaid expansion 2015-01-01) and TX (no expansion)
# both NULL, in an identifiable regime, at the tightest delta in the map
# (PREREG_ACS_EXTENSION S10, falsification branch S6(b)). The pre-registered mechanism for that
# miss -- written down in S5 as risk #1, before the run -- is that AUC is rank-based and the
# expansion moved an eligibility THRESHOLD without reordering anyone.
#
# A1 tests exactly that, and it does NOT touch the instrument: src/analysis/drift_measure.py
# already carries concept_within_overlap_multi, which returns the within-overlap gap under
# {auc, brier, logloss, kl}. Brier and log-loss are proper scores and are NOT rank-invariant, so
# if the mechanism is right the AUC gap stays ~0 while Brier/log-loss go positive. A different
# lens on the same rows -> the PREREG S4 battery gate is not reopened.
#
# Predictions were committed before this ran (PREREG_ACS_EXTENSION S10.6 and the session record):
#   A1 PA brier/logloss > placebo 60% | A1 PA auc ~ placebo 80% | A1 TX all ~ placebo 75%
#   A2 PA temporal ~ placebo 60% (that lens is also AUC-based) | A2 PA-TX spatial > placebo 65%
#   A3 max-n unchanged 85% | A3 class panel all null 70%
# If A1 is negative too, the blind spot is NOT the metric and S6.4 gets harder. Report either way.
set -u
cd ~/explainableTabular
PY=$HOME/miniconda3/envs/explaintab311/bin/python
CFG=configs/phase1.yaml
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
mkdir -p logs

$PY -c "import sys,sklearn;v=(sys.version.split()[0],sklearn.__version__);print('ENV',v);assert v==('3.11.15','1.9.0');print('ENV OK')" || { echo "### ENV FAIL"; exit 1; }

# ── Build the clean pre/post parquets ────────────────────────────────────────
# run_gap_hygiene splits early/late at the MEDIAN of t. On the full 2014-2018 panel the median
# lands in 2016, which buries PA's 2015 expansion inside the "early" half and dilutes the very
# contrast being measured. Restricting to {2014, 2018} makes the median split identical to the
# documented pre/post boundary, and is also the pair run_whyshift uses by default.
$PY - << 'EOF'
import pandas as pd
for s in ("PA", "TX"):
    df = pd.read_parquet(f"acs_pubcov_{s}.parquet")
    sub = df[df["YEAR"].isin([2014, 2018])].reset_index(drop=True)
    sub.to_parquet(f"acs_pubcov_{s}_1418.parquet")
    print(f"  {s} 2014+2018: n={len(sub)}  " +
          "  ".join(f"{y}:{sub[sub.YEAR==y]['label'].mean():.3f}" for y in (2014, 2018)), flush=True)
EOF
echo "### PARQUETS READY $(date -u +%FT%TZ)"

# ── A1. multi-metric within-overlap lens (the decisive one) ──────────────────
for S in PA TX; do
  $PY scripts/run_gap_hygiene.py --csv acs_pubcov_${S}_1418.parquet --target label --time YEAR \
      --name acs_pubcov_${S}_1418 --n-seeds 15 --cuts 0.2 0.35 0.5 0.65 0.8 \
      2>&1 | tee logs/a1_gaphyg_${S}.log
done
echo "### A1 DONE $(date -u +%FT%TZ)"

# ── A2. WhyShift spatial-vs-temporal on the same task (zero new code) ────────
# Its lens is also AUC-based, so it is a check on WHICH lens rather than which metric; the
# spatial pair doubles as the WhyShift contrast this project claimed but never ran on pubcov.
$PY scripts/run_whyshift.py --states PA TX --years 2014 2018 --task pubcov \
    2>&1 | tee logs/a2_whyshift_pubcov.log
echo "### A2 DONE $(date -u +%FT%TZ)"

# ── A3. loose ends, deployment lens, no code ────────────────────────────────
# (a) is the null an artifact of the default 1-in-3 thinning? delta was 0.00083, so probably not.
$PY scripts/run_deployment_decay.py --csv acs_pubcov_PA.parquet --target label --time YEAR \
    --by-value --n-seeds 10 --max-n 250000 --name acs_pubcov_PA_fullN 2>&1 | tee logs/a3_maxn_PA.log
# (b) does a different probe class see the expansion? (S6.1 material)
for M in rf linear; do
  for S in PA TX; do
    $PY scripts/run_deployment_decay.py --csv acs_pubcov_${S}.parquet --target label --time YEAR \
        --by-value --n-seeds 10 --model $M --name acs_pubcov_${S}_${M} 2>&1 | tee logs/a3_${M}_${S}.log
  done
done
# (c) full-span family sweep was exploratory only; weather_fullspan's promotion needs confirmatory.
for FC in "lowvar lo" "interaction lo" "subpop lo"; do set -- $FC
  $PY scripts/run_deployment_decay.py --tabred homecredit_default weather homesite_insurance \
      --config $CFG --n-seeds 10 --tabred-span full --seed-base 100 \
      --inj-family $1 --inj-cols $2 2>&1 | tee logs/a3_fsconf_$1_$2.log
done
echo "### A3 DONE $(date -u +%FT%TZ)"

echo "=== 완료 점검 $(date -u +%FT%TZ) ==="
for f in logs/a3_maxn_PA.log:1 logs/a3_rf_PA.log:1 logs/a3_rf_TX.log:1 \
         logs/a3_linear_PA.log:1 logs/a3_linear_TX.log:1 \
         logs/a3_fsconf_lowvar_lo.log:3 logs/a3_fsconf_interaction_lo.log:3 \
         logs/a3_fsconf_subpop_lo.log:3; do
  p=${f%:*}; want=${f#*:}; n=$(grep -c "=> " "$p" 2>/dev/null || echo 0)
  [ "$n" -eq "$want" ] && echo "OK   $p  판정 $n/$want" || echo "FAIL $p  판정 $n/$want  <<<"
done
for p in logs/a1_gaphyg_PA.log logs/a1_gaphyg_TX.log logs/a2_whyshift_pubcov.log; do
  n=$(grep -c "brier\|cov_auc" "$p" 2>/dev/null || echo 0)
  [ "$n" -gt 0 ] && echo "OK   $p  ($n 지표 줄)" || echo "FAIL $p  비어 있음  <<<"
done
