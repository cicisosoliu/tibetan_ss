#!/usr/bin/env bash
# Evaluate a trained checkpoint on the test split.
# Usage: scripts/evaluate.sh configs/experiment/proposed.yaml outputs/logs/proposed/checkpoints/last.ckpt
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="$1"
CKPT="$2"; shift 2 || true
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python -m tibetan_ss.cli.evaluate --config "$CONFIG" --checkpoint "$CKPT" "$@"
