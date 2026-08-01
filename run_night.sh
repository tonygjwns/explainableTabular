#!/usr/bin/env bash
set -u
cd ~/explainableTabular
PY=$HOME/miniconda3/envs/explaintab311/bin/python
CFG=configs/phase1.yaml
TAB="cooking_time delivery_eta ecom_offers homecredit_default weather homesite_insurance"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
mkdir -p logs

$PY -c "import sys,sklearn;v=(sys.version.split()[0],sklearn.__version__);print('ENV',v);assert v==('3.11.15','1.9.0');print('ENV OK')" || { echo "### ENV FAIL"; exit 1; }

$PY scripts/smoke_test_inj_family.py 2>&1 | tee logs/smoke_mapenv.log
grep -q "SMOKE PASS" logs/smoke_mapenv.log || { echo "### SMOKE FAIL"; exit 1; }

$PY scripts/run_deployment_decay.py --synth 2>&1 | tee logs/synth_mapenv.log
grep -q "GROUND-TRUTH PASS" logs/synth_mapenv.log || { echo "### BATTERY FAIL"; exit 1; }
cp results/phase1/deployment_decay/synth_summary.json \
   results/phase1/deployment_decay/synth_battery_PASS_mapenv_$(git rev-parse --short HEAD).json
echo "### GATE PASSED $(date -u +%FT%TZ)"

for FC in "topvar hi" "lowvar lo" "interaction lo" "subpop lo"; do set -- $FC
  $PY scripts/run_deployment_decay.py --river agrawal_abrupt --n-seeds 10 \
      --inj-family $1 --inj-cols $2 2>&1 | tee logs/ctrl_$1_$2.log
done
echo "### CONTROLS DONE $(date -u +%FT%TZ)"

$PY scripts/run_deployment_decay.py --tabred weather ecom_offers sberbank_housing --config $CFG --n-seeds 10 --tabred-span full > logs/fs_A.log 2>&1 &
sleep 420
$PY scripts/run_deployment_decay.py --tabred delivery_eta homecredit_default --config $CFG --n-seeds 10 --tabred-span full > logs/fs_B.log 2>&1 &
sleep 420
$PY scripts/run_deployment_decay.py --tabred cooking_time homesite_insurance maps_routing --config $CFG --n-seeds 10 --tabred-span full > logs/fs_C.log 2>&1 &
sleep 420
for FC in "topvar hi" "lowvar lo" "interaction lo" "subpop lo"; do set -- $FC
  F=$1; C=$2
  ( $PY scripts/run_deployment_decay.py --tabred $TAB --config $CFG --n-seeds 10 --inj-family $F --inj-cols $C  > logs/d1_${F}_${C}.log 2>&1
    $PY scripts/run_deployment_decay.py --elec2   --n-seeds 10 --inj-family $F --inj-cols $C >> logs/d1_${F}_${C}.log 2>&1
    $PY scripts/run_deployment_decay.py --insects --n-seeds 10 --inj-family $F --inj-cols $C >> logs/d1_${F}_${C}.log 2>&1 ) &
  sleep 420
done
wait

echo "=== 완료 점검 $(date -u +%FT%TZ) ==="
for f in logs/fs_A.log:3 logs/fs_B.log:2 logs/fs_C.log:3 \
         logs/d1_topvar_hi.log:8 logs/d1_lowvar_lo.log:8 \
         logs/d1_interaction_lo.log:8 logs/d1_subpop_lo.log:8; do
  p=${f%:*}; want=${f#*:}; n=$(grep -c "=> " "$p" 2>/dev/null || echo 0)
  [ "$n" -eq "$want" ] && echo "OK   $p  판정 $n/$want" || echo "FAIL $p  판정 $n/$want  <<<"
done
