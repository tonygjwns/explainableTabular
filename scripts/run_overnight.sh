#!/usr/bin/env bash
# One-command overnight job: preprocess all TabReD data, then run Phase 0 on all 8.
# Logs everything to logs/overnight_<timestamp>.log and is safe under nohup.
#
# Usage (from repo root):
#   ln -s ~/external/tabred/data data        # once, if not already linked
#   nohup bash scripts/run_overnight.sh > /dev/null 2>&1 &
#   tail -f logs/overnight_*.log              # watch progress
#
# Env:
#   TABRED_REPO  (default ~/external/tabred)
#   SKIP_PREP=1  to skip data preprocessing (data already generated)

set -u
TABRED_REPO="${TABRED_REPO:-$HOME/external/tabred}"
mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/overnight_${TS}.log"

echo "logging to $LOG"
{
  echo "######## overnight run $TS ########"
  echo "repo: $(pwd)  | tabred: $TABRED_REPO"

  if [ "${SKIP_PREP:-0}" != "1" ]; then
    echo "==== STEP 1/2: preprocess all TabReD data ===="
    TABRED_REPO="$TABRED_REPO" bash scripts/prepare_all_data.sh
  else
    echo "==== STEP 1/2: SKIPPED (SKIP_PREP=1) ===="
  fi

  echo "==== STEP 2/2: Phase 0 (all 8; missing data auto-skipped) ===="
  python scripts/run_phase0.py --config configs/tabm_baseline.yaml

  echo "######## done $TS ########"
} 2>&1 | tee -a "$LOG"
