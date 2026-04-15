#!/usr/bin/env bash
# Unified launcher. Usage:
#   scripts/train.sh configs/experiment/proposed.yaml
#   scripts/train.sh configs/experiment/baseline_tiger.yaml training.dataloader.batch_size=8
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
CONFIG="$1"; shift || true
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

python -m tibetan_ss.cli.train --config "$CONFIG" "$@"
