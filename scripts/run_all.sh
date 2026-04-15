#!/usr/bin/env bash
# Train every experiment sequentially. Assumes data has already been prepared.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

EXPERIMENTS=(
    "baseline_tiger"
    "baseline_sepreformer"
    "baseline_dual_path_mamba"
    "baseline_mossformer2"
    "ext_dip"
    "proposed"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo ">>> training $exp"
    python -m tibetan_ss.cli.train --config "$REPO_ROOT/configs/experiment/${exp}.yaml"
done

python -m tibetan_ss.cli.aggregate_results \
    --root "$REPO_ROOT/outputs/logs" \
    --output "$REPO_ROOT/outputs/summary.md"
