#!/usr/bin/env bash
# Day-4 queue: the three experiments still owed, plus the anchor sweep that completes the
# sensitivity profile. Predictions are committed in this header BEFORE the runs.
#
# The instrument gained two OPT-IN diagnostic flags (--metric, --mi-k), both defaulting to the
# current behaviour, so PREREG S4 requires a battery re-gate and the battery must come back
# BIT-IDENTICAL -- if it does not, the change is reverted rather than explained.
#
# ── E1  Does a proper score see what the rank-based one missed? ──────────────────────────────
#   PREREG_ACS_EXTENSION S11 established, through a DIFFERENT lens, that the treatment/control
#   separation widens 3.1x (AUC) -> 6.3x (Brier) -> 6.6x (log-loss). E1 asks the same question of
#   THE PROBE ITSELF on ACS: does staleness_harm fire on PA under Brier/log-loss?
#   DIAGNOSTIC ONLY -- changing the score changes the estimand and voids the AUC-calibrated floor,
#   so these runs are never map verdicts (S10.6).
#   PREDICTION: PA staleness turns positive under brier/logloss 55%; clears the 0.02 floor 25%;
#               TX stays <= 0 under every metric 70%.
#   READING RULE (fixed before the runs): read staleness_harm / denoised_staleness, NOT the
#   verdict label. The cascade's floor, noise gate and envelope are all calibrated in AUC units,
#   so under a proper score the labels are meaningless by construction -- a local check already
#   returns INCONCLUSIVE on a cell whose rule genuinely rotates (staleness +0.214). Any verdict
#   emitted by an E1 run is discarded; only the arm magnitudes and the PA-vs-TX contrast are read.
#
# ── E3  Does D itself carry the power collapse, or is it dataset identity? ───────────────────
#   Across the panel, recovery of a fixed-strength planted rule falls as D rises -- but every
#   comparison there is between different datasets. --mi-k narrows the representation of ONE cell,
#   moving D while holding the cell fixed. Run on the two highest-D certified cells.
#   PREDICTION: D falls monotonically with k 80%; injection recovery RISES as D falls on
#               delivery_eta 60%; the relation holds within-cell at all 50%.
#
# ── E2  Do type-attributing frames separate rule change from noise decay? ────────────────────
#   The abstract's closing claim currently rests on our own instrument and our own canaries.
#   This points DISDE-style reweighting health and the within-overlap decomposition at battery
#   cells whose ground truth is fixed by construction. Under the field-standard definition
#   (Webb/Gama) noise decay IS a Y|X change, so a fire there is not an error -- the question is
#   whether the frames SEPARATE the two mechanisms by magnitude.
#   PREDICTION: rule cells exceed noise cells by >5x 65%; noise cells still show a positive
#               Y|X gap (i.e. sign alone does not separate) 75%.
#
# ── E4  Anchor sweep across families (completes the S6.3 sensitivity profile) ────────────────
#   PREDICTION: all four combinations recover on every single-switch anchor 70%.
set -u
cd ~/explainableTabular
PY=$HOME/miniconda3/envs/explaintab311/bin/python
CFG=configs/phase1.yaml
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
mkdir -p logs

$PY -c "import sys,sklearn;v=(sys.version.split()[0],sklearn.__version__);print('ENV',v);assert v==('3.11.15','1.9.0');print('ENV OK')" || { echo "### ENV FAIL"; exit 1; }

# ── gate: the two new flags must leave the battery bit-identical ─────────────────────────────
$PY scripts/smoke_test_inj_family.py 2>&1 | tee logs/d4_smoke.log
grep -q "SMOKE PASS" logs/d4_smoke.log || { echo "### SMOKE FAIL"; exit 1; }
$PY scripts/run_deployment_decay.py --synth 2>&1 | tee logs/d4_synth.log
grep -q "GROUND-TRUTH PASS" logs/d4_synth.log || { echo "### BATTERY FAIL"; exit 1; }
cp results/phase1/deployment_decay/synth_summary.json \
   results/phase1/deployment_decay/synth_battery_PASS_$(git rev-parse --short HEAD).json
echo "### GATE PASSED $(date -u +%FT%TZ)"

# ── E1 ──────────────────────────────────────────────────────────────────────────────────────
for M in brier logloss; do
  for S in PA TX; do
    $PY scripts/run_deployment_decay.py --csv acs_pubcov_${S}.parquet --target label --time YEAR \
        --by-value --n-seeds 10 --metric $M --name acs_pubcov_${S}_${M} 2>&1 | tee logs/e1_${M}_${S}.log
  done
done
# the two industrial binary cells that never earned a certificate, under a proper score
for M in brier logloss; do
  $PY scripts/run_deployment_decay.py --tabred ecom_offers homecredit_default homesite_insurance \
      --config $CFG --n-seeds 10 --metric $M 2>&1 | tee logs/e1_tabred_${M}.log
done
echo "### E1 DONE $(date -u +%FT%TZ)"

# ── E3 ──────────────────────────────────────────────────────────────────────────────────────
for K in 5 10 20 50; do
  $PY scripts/run_deployment_decay.py --tabred delivery_eta homecredit_default \
      --config $CFG --n-seeds 10 --mi-k $K 2>&1 | tee logs/e3_k${K}.log
done
echo "### E3 DONE $(date -u +%FT%TZ)"

# ── E2 ──────────────────────────────────────────────────────────────────────────────────────
$PY scripts/run_detector_headtohead.py --n-seeds 10 2>&1 | tee logs/e2_h2h.log
echo "### E2 DONE $(date -u +%FT%TZ)"

# ── E4 ──────────────────────────────────────────────────────────────────────────────────────
for FC in "topvar hi" "lowvar lo" "interaction lo" "subpop lo"; do set -- $FC
  $PY scripts/run_deployment_decay.py --river stagger_abrupt sine_abrupt hyperplane_incr \
      --n-seeds 10 --inj-family $1 --inj-cols $2 2>&1 | tee logs/e4_$1_$2.log
done
echo "### E4 DONE $(date -u +%FT%TZ)"

echo "=== 완료 점검 $(date -u +%FT%TZ) ==="
for f in logs/e1_brier_PA.log:1 logs/e1_brier_TX.log:1 logs/e1_logloss_PA.log:1 \
         logs/e1_logloss_TX.log:1 logs/e1_tabred_brier.log:3 logs/e1_tabred_logloss.log:3 \
         logs/e3_k5.log:2 logs/e3_k10.log:2 logs/e3_k20.log:2 logs/e3_k50.log:2 \
         logs/e4_topvar_hi.log:3 logs/e4_lowvar_lo.log:3 \
         logs/e4_interaction_lo.log:3 logs/e4_subpop_lo.log:3; do
  p=${f%:*}; want=${f#*:}; n=$(grep -c "=> " "$p" 2>/dev/null); n=${n:-0}
  [ "$n" -eq "$want" ] && echo "OK   $p  판정 $n/$want" || echo "FAIL $p  판정 $n/$want  <<<"
done
n=$(grep -c "synth_" logs/e2_h2h.log 2>/dev/null); n=${n:-0}
[ "$n" -ge 6 ] && echo "OK   logs/e2_h2h.log  셀 $n" || echo "FAIL logs/e2_h2h.log  셀 $n  <<<"
