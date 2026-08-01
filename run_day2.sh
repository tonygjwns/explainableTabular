#!/usr/bin/env bash
# Day-2 server queue: A (ACS natural experiment) + B (full-span hardening) + C (panel completion).
#
# Reading rules for every cell below were committed BEFORE this script ran:
#   A -> PREREG_ACS_EXTENSION_2026-07-31.md  (commit 3332a14, 2026-07-31)
#   B -> PREREG_DEPLOYMENT_V2.md S14 [C] + S16
#   C -> PREREG_DEPLOYMENT_V2.md S16
# Nothing here changes a threshold, the cascade, or the seed protocol.
#
# The instrument is UNCHANGED since the S4 battery gate passed at 04:29Z on 2026-08-01
# (map env 3.11.15/1.9.0, 14/14). Only scripts/prep_folktables.py changed, which is a data
# materializer and never touches the probe -- so the gate is not reopened.
set -u
cd ~/explainableTabular
PY=$HOME/miniconda3/envs/explaintab311/bin/python
CFG=configs/phase1.yaml
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
mkdir -p logs

$PY -c "import sys,sklearn;v=(sys.version.split()[0],sklearn.__version__);print('ENV',v);assert v==('3.11.15','1.9.0');print('ENV OK')" || { echo "### ENV FAIL"; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# A. ACS PublicCoverage natural experiment.
#    PA implemented ACA Medicaid expansion 2015-01-01 (Healthy PA waiver, CMS approved
#    2014-08-28), so within the 2014-2018 window 2014 is pre and 2015+ is post -- the old
#    anchor sits on the far side of a documented rule change. TX never expanded through 2018.
#    Same task, same years, same instrument: one has an externally documented change in
#    P(y|x), the other does not.
#    STOP POINT (PREREG_ACS_EXTENSION S9-3): read PA and TX before running any further state.
# ─────────────────────────────────────────────────────────────────────────────
echo "### A: ACS PREP START $(date -u +%FT%TZ)"
for S in PA TX; do
  [ -f acs_pubcov_$S.parquet ] || $PY scripts/prep_folktables.py $S acs_pubcov_$S.parquet pubcov 2>&1 | tee logs/acsprep_$S.log
done
for S in PA TX; do
  [ -f acs_pubcov_$S.parquet ] || { echo "### A: PREP FAILED for $S — skipping ACS"; SKIP_A=1; }
done
if [ -z "${SKIP_A:-}" ]; then
  for S in PA TX; do
    $PY scripts/run_deployment_decay.py --csv acs_pubcov_$S.parquet --target label --time YEAR \
        --by-value --n-seeds 10 --name acs_pubcov_$S 2>&1 | tee logs/acs_$S.log
  done
  echo "### A DONE $(date -u +%FT%TZ)"
else
  echo "### A SKIPPED $(date -u +%FT%TZ)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# B + C, launched in parallel with 7-minute stagger.
#   B1  full-span confirmatory (seeds 100-109). S14 did not require it (no CONCEPT fired),
#       but the full-span read moved two cells relative to the train-span map
#       (sberbank NOISE-DRIFT-CONFOUNDED -> UNIDENT, cooking INJECTION-RECOVERED ->
#       NO-STRONG-CONCEPT as its D fell under D*), and an appendix claim needs the same
#       stability evidence the main map has.
#   B2  family x carrier sweep ON the full-span cells. homecredit_default_fullspan reads
#       denoised +0.0188 -- 94% of the decision floor, the largest live positive-direction
#       number anywhere in the panel -- and its certificate came back EARNED rather than
#       vacuous. Whether that certificate survives a change of signal family is exactly the
#       question S16 exists to ask.
#   C   sberbank_housing into the [D1] sweep. It was outside the S14 cell list because its
#       primary verdict routes to the noise branch, but its STRICT shadow lands
#       UNIDENTIFIABLE and therefore reaches the injection stage (S15b). Completing it makes
#       the sweep cover the panel rather than six of eight cells.
# ─────────────────────────────────────────────────────────────────────────────
echo "### B+C LAUNCH $(date -u +%FT%TZ)"

$PY scripts/run_deployment_decay.py --tabred weather ecom_offers sberbank_housing \
    --config $CFG --n-seeds 10 --seed-base 100 --tabred-span full > logs/fsc_A.log 2>&1 &
sleep 420
$PY scripts/run_deployment_decay.py --tabred delivery_eta homecredit_default \
    --config $CFG --n-seeds 10 --seed-base 100 --tabred-span full > logs/fsc_B.log 2>&1 &
sleep 420
$PY scripts/run_deployment_decay.py --tabred cooking_time homesite_insurance maps_routing \
    --config $CFG --n-seeds 10 --seed-base 100 --tabred-span full > logs/fsc_C.log 2>&1 &
sleep 420

for FC in "lowvar lo" "interaction lo" "subpop lo"; do set -- $FC
  F=$1; C=$2
  $PY scripts/run_deployment_decay.py --tabred homecredit_default weather homesite_insurance \
      --config $CFG --n-seeds 10 --tabred-span full --inj-family $F --inj-cols $C \
      > logs/fsd1_${F}_${C}.log 2>&1 &
  sleep 420
done

for FC in "topvar hi" "lowvar lo" "interaction lo" "subpop lo"; do set -- $FC
  F=$1; C=$2
  $PY scripts/run_deployment_decay.py --tabred sberbank_housing --config $CFG --n-seeds 10 \
      --inj-family $F --inj-cols $C > logs/sb_${F}_${C}.log 2>&1 &
  sleep 180
done
wait

echo "=== 완료 점검 $(date -u +%FT%TZ) ==="
for f in logs/acs_PA.log:1 logs/acs_TX.log:1 \
         logs/fsc_A.log:3 logs/fsc_B.log:2 logs/fsc_C.log:3 \
         logs/fsd1_lowvar_lo.log:3 logs/fsd1_interaction_lo.log:3 logs/fsd1_subpop_lo.log:3 \
         logs/sb_topvar_hi.log:1 logs/sb_lowvar_lo.log:1 \
         logs/sb_interaction_lo.log:1 logs/sb_subpop_lo.log:1; do
  p=${f%:*}; want=${f#*:}; n=$(grep -c "=> " "$p" 2>/dev/null || echo 0)
  [ "$n" -eq "$want" ] && echo "OK   $p  판정 $n/$want" || echo "FAIL $p  판정 $n/$want  <<<"
done
