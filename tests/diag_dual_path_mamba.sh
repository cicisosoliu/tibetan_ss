#!/usr/bin/env bash
# Diagnostic: step-by-step Dual-path Mamba construction + forward.
# Usage:  cd /hy-tmp/tibetan_ss && bash tests/diag_dual_path_mamba.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

timeout 120 python3 << 'PYEOF'
import torch, sys
print("1. importing adapter...", flush=True)
from tibetan_ss.models.dual_path_mamba import DualPathMambaAdapter
print("2. building model (small: n_dp=4, n_encoder=128)...", flush=True)
model = DualPathMambaAdapter(
    sample_rate=16000, num_speakers=2,
    n_encoder_out=128, kernel_size=32, n_dp=4, chunk_size=200,
    n_mamba_dp=2, d_state=16, mamba_expand=2, mamba_conv=4,
    bidirectional=True, skip_around_intra=True,
)
print("3. model params:", sum(p.numel() for p in model.parameters()), flush=True)
print("4. moving to cuda...", flush=True)
model = model.cuda()
print("5. forward...", flush=True)
x = torch.randn(1, 48000).cuda()
y = model(x)
print("6. output:", y.shape, flush=True)
assert y.shape == (1, 2, 48000), f"unexpected shape {y.shape}"
print("ALL DONE", flush=True)
PYEOF
echo "exit code: $?"
