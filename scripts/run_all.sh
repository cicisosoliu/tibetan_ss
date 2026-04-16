#!/usr/bin/env bash
# Train every experiment sequentially. Assumes data has already been prepared.
#
# Usage:
#   scripts/run_all.sh                 # NICT-Tib1 + DEMAND formal runs (default)
#   scripts/run_all.sh public          # original public-dataset experiments
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

MODE="${1:-nict}"

if [[ "$MODE" == "nict" ]]; then
    EXPERIMENTS=(
        "baseline_tiger_nict"
        "baseline_sepreformer_nict"
        "baseline_dual_path_mamba_nict"
        "baseline_mossformer2_nict"
        "ext_dip_nict"
        "proposed_formal"
    )
elif [[ "$MODE" == "public" ]]; then
    EXPERIMENTS=(
        "baseline_tiger"
        "baseline_sepreformer"
        "baseline_dual_path_mamba"
        "baseline_mossformer2"
        "ext_dip"
        "proposed"
    )
else
    echo "unknown mode: $MODE (expected nict|public)" >&2
    exit 1
fi

for exp in "${EXPERIMENTS[@]}"; do
    echo ">>> training $exp"
    python -m tibetan_ss.cli.train --config "$REPO_ROOT/configs/experiment/${exp}.yaml"
done

python -m tibetan_ss.cli.aggregate_results \
    --root "$REPO_ROOT/outputs/logs" \
    --output "$REPO_ROOT/outputs/summary.md"
