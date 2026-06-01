#!/usr/bin/env bash
# Preprocess all 8 TabReD datasets into <tabred>/data via their Kaggle scripts.
# Continues on failure (e.g. a competition whose rules aren't accepted yet) and
# reports a summary at the end. Idempotent-ish: skips a dataset if its output exists.
#
# Usage:
#   TABRED_REPO=~/external/tabred bash scripts/prepare_all_data.sh
# Prereqs (SETUP.md §3): kaggle auth working; competition rules accepted for
#   homesite / ecom-offers / homecredit; polars==0.20.19 etc. installed.

set -u
TABRED_REPO="${TABRED_REPO:-$HOME/external/tabred}"

if [ ! -d "$TABRED_REPO/preprocessing" ]; then
  echo "ERROR: TabReD repo not found at $TABRED_REPO (set TABRED_REPO=...)." >&2
  exit 1
fi

cd "$TABRED_REPO" || exit 1
mkdir -p data

# folder name (as written on disk) -> preprocessing script (same hyphenated name)
SCRIPTS=(
  sberbank-housing
  cooking-time
  delivery-eta
  maps-routing
  weather
  homesite
  ecom-offers
  homecredit
)

ok=(); fail=(); skip=()
for s in "${SCRIPTS[@]}"; do
  if [ -f "data/$s/X_meta.npy" ]; then
    echo "=== [skip] $s (already present) ==="; skip+=("$s"); continue
  fi
  echo "=== [run] $s ==="
  if PYTHONPATH=. python "preprocessing/$s.py"; then
    ok+=("$s")
  else
    echo "!!! [fail] $s (see error above; competition rules accepted?)" >&2
    fail+=("$s")
  fi
done

echo
echo "==== preprocessing summary ===="
echo "ok   : ${ok[*]:-(none)}"
echo "skip : ${skip[*]:-(none)}"
echo "fail : ${fail[*]:-(none)}"
