#!/usr/bin/env bash
# One-shot data preparation: scan NICT-Tib1 + DEMAND, then generate mixtures
# for all three splits. Override TIBETAN_ROOT, DEMAND_ROOT,
# TIBETAN_SS_OUTPUT via env or .env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="${1:-$REPO_ROOT/configs/data/sr16k.yaml}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python -m tibetan_ss.data.scripts.prepare_nict_tib1 \
    --config "$CONFIG"

python -m tibetan_ss.data.scripts.generate_mixtures \
    --config "$CONFIG" \
    --splits train val test
