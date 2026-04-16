#!/usr/bin/env bash
# End-to-end smoke test for the NICT-Tib1 local dataset with dynamic mixing.
#
# What this script does:
#   1. Uses configs/data/nict_tib1.yaml (noise disabled, DM on).
#   2. Runs prepare_nict_tib1.py → writes speaker/noise manifests.
#   3. Runs generate_mixtures.py → only produces 200 val + 200 test clips
#      (train is dynamic, no offline generation).
#   4. Launches the proposed model for 3 epochs with a shrunken architecture.
#
# Usage (from tibetan_ss/ root):
#   scripts/smoke_test.sh                        # default paths
#   TIBETAN_ROOT=/custom/path scripts/smoke_test.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

: "${TIBETAN_ROOT:=$REPO_ROOT/Tibetan}"
: "${TIBETAN_SS_OUTPUT:=$REPO_ROOT/data}"
: "${DEMAND_ROOT:=}"                             # leave empty → noise disabled
export TIBETAN_ROOT TIBETAN_SS_OUTPUT DEMAND_ROOT

DATA_CFG="$REPO_ROOT/configs/data/nict_tib1.yaml"
EXP_CFG="$REPO_ROOT/configs/experiment/smoke_proposed.yaml"

echo "==========================================================="
echo " Tibetan speech separation — NICT-Tib1 smoke test"
echo " TIBETAN_ROOT      = $TIBETAN_ROOT"
echo " TIBETAN_SS_OUTPUT = $TIBETAN_SS_OUTPUT"
echo " DEMAND_ROOT       = ${DEMAND_ROOT:-<unset, noise disabled>}"
echo "==========================================================="

echo ">>> [1/3] prepare speaker / (empty) noise manifests"
python -m tibetan_ss.data.scripts.prepare_nict_tib1 \
    --config "$DATA_CFG"

echo ">>> [2/3] generate tiny offline val/test mixtures (train is DM-only)"
python -m tibetan_ss.data.scripts.generate_mixtures \
    --config "$DATA_CFG" \
    --splits train val test

echo ">>> [3/3] run 3-epoch training with dynamic mixing"
python -m tibetan_ss.cli.train --config "$EXP_CFG" "$@"

echo "==========================================================="
echo " smoke test finished"
echo " logs:         outputs/logs/smoke_proposed/"
echo " checkpoints:  outputs/logs/smoke_proposed/checkpoints/"
echo "==========================================================="
