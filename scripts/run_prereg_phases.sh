#!/usr/bin/env bash
# PREREG_DEPLOYMENT_V2 §5 — Phase 1~4 one-shot driver (server, env explaintab311).
#   nohup bash scripts/run_prereg_phases.sh > prereg_run.log 2>&1 &
# Resumable: each step drops a marker in results/phase1/deployment_decay/markers/ and is skipped
# on re-run. A failing step logs and continues (per-dataset failures are already absorbed by the
# instrument itself). Phase 1 (sberbank K-sweep) is included for completeness; if already run,
# markers can be pre-created: touch markers/p1_K{5,8,10,12,20}.done
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
CFG=configs/phase1.yaml
OUT=${OUT_DIR:-results/phase1/deployment_decay}
MARK=$OUT/markers
PHASES=${PHASES:-1234}    # e.g. PHASES=2 to run only Phase 2 (enables parallel shells per phase)
want () { case "$PHASES" in *"$1"*) return 0;; *) return 1;; esac; }
mkdir -p "$MARK"
TABRED="sberbank_housing homesite_insurance ecom_offers homecredit_default cooking_time delivery_eta maps_routing weather"
EMBER_PARQUET=${EMBER_PARQUET:-data/ember/ember.parquet}

step () {  # step <marker> <cmd...>
  local m="$MARK/$1.done"; shift
  if [ -f "$m" ]; then echo "[skip] $m"; return 0; fi
  echo "[run ] $* ($(date -u +%FT%TZ))"
  if "$@"; then touch "$m"; else echo "[FAIL] $* — continuing" >&2; fi
}

want 1 && {
echo "==== PREREG Phase 1: sberbank K-sweep (decisive read) ===="
for K in 5 8 10 12 20; do
  step "p1_K$K" $PY scripts/run_deployment_decay.py --tabred sberbank_housing --config $CFG \
    --n-seeds 10 --windows $K
done

}
want 2 && {
echo "==== PREREG Phase 2a: full map, exploratory seeds 0-9 ===="
step "p2a_tabred"  $PY scripts/run_deployment_decay.py --tabred $TABRED --config $CFG --n-seeds 10
step "p2a_elec2"   $PY scripts/run_deployment_decay.py --elec2 --n-seeds 10
step "p2a_insects" $PY scripts/run_deployment_decay.py --insects --n-seeds 10

echo "==== PREREG Phase 2b: confirmatory rerun, seeds 100-109 (report verdict moves as unstable) ===="
step "p2b_tabred"  $PY scripts/run_deployment_decay.py --tabred $TABRED --config $CFG --n-seeds 10 --seed-base 100
step "p2b_elec2"   $PY scripts/run_deployment_decay.py --elec2 --n-seeds 10 --seed-base 100
step "p2b_insects" $PY scripts/run_deployment_decay.py --insects --n-seeds 10 --seed-base 100

}
want 3 && {
echo "==== PREREG Phase 3: model-class panel (rf decision-grade; linear/knn canaries) ===="
for M in rf linear knn; do
  step "p3_${M}_tabred"  $PY scripts/run_deployment_decay.py --tabred $TABRED --config $CFG --n-seeds 10 --model $M
  step "p3_${M}_elec2"   $PY scripts/run_deployment_decay.py --elec2 --n-seeds 10 --model $M
  step "p3_${M}_insects" $PY scripts/run_deployment_decay.py --insects --n-seeds 10 --model $M
done

}
want 4 && {
echo "==== PREREG Phase 4: anchors (designed drift MUST fire; else instrument defect, hold the map) ===="
step "p4_river"   $PY scripts/run_deployment_decay.py --river all --n-seeds 10
step "p4_insects" $PY scripts/run_deployment_decay.py --insects --insects-variant all --n-seeds 10
if [ -f "$EMBER_PARQUET" ]; then
  step "p4_ember" $PY scripts/run_deployment_decay.py --csv "$EMBER_PARQUET" --target label \
    --time appeared --by-value --n-seeds 10 --name ember
else
  echo "[note] EMBER parquet not found at $EMBER_PARQUET (set EMBER_PARQUET=...); skipping p4_ember"
fi

}
pip freeze > env_explaintab311_freeze.txt || true
echo "==== DONE ($(date -u +%FT%TZ)) — send results/phase1/deployment_decay/summary_*.json + env freeze ===="
