#!/usr/bin/env bash
# Compute params / MACs / RTF for all 6 models and print a comparison table.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

DEVICE="${1:-cuda}"

EXPERIMENTS=(
    "proposed_formal"
    "baseline_tiger_nict"
    "baseline_sepreformer_nict"
    "baseline_dual_path_mamba_nict"
    "baseline_mossformer2_nict"
    "ext_dip_nict"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo ">>> $exp"
    python -m tibetan_ss.cli.model_complexity \
        --config "$REPO_ROOT/configs/experiment/${exp}.yaml" \
        --duration 3.0 --device "$DEVICE" || true
    echo
done
