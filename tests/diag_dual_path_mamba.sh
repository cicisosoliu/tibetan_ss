#!/usr/bin/env bash
# Diagnostic: verify upstream BiMamba works with causal_conv1d compat shim.
# Usage:  cd /hy-tmp/tibetan_ss && bash tests/diag_dual_path_mamba.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

timeout 120 python3 << 'PYEOF'
import torch, sys
print("1. testing compat shim install...", flush=True)
from tibetan_ss.models._causal_conv1d_compat import install_compat_shim
install_compat_shim()
print("2. shim OK", flush=True)

print("3. importing upstream MambaBlocksSequential...", flush=True)
sys.path.insert(0, "third_party/Mamba-TasNet")
import modules.mamba_blocks as _mb
if _mb.RMSNorm is None:
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
        _mb.RMSNorm = RMSNorm
    except ImportError:
        import torch.nn as nn
        class _RMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6, **kwargs):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.eps = eps
            def forward(self, x):
                xf = x.float()
                return (self.weight * xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        _mb.RMSNorm = _RMSNorm
from modules.mamba_blocks import MambaBlocksSequential
print("4. building upstream BiMamba stack...", flush=True)
stack = MambaBlocksSequential(
    n_mamba=1, bidirectional=True, d_model=128, d_state=16,
    expand=2, d_conv=4, fused_add_norm=False, rms_norm=True,
    residual_in_fp32=False, conv_bias=True, bias=False,
).cuda()
print("5. forward...", flush=True)
x = torch.randn(2, 50, 128).cuda()
y = stack(x)
print("6. output:", y.shape, flush=True)
print("7. backward...", flush=True)
y.sum().backward()
print("8. backward OK", flush=True)

print("9. full adapter test...", flush=True)
from tibetan_ss.models.dual_path_mamba import DualPathMambaAdapter
model = DualPathMambaAdapter(
    sample_rate=16000, num_speakers=2,
    n_encoder_out=128, kernel_size=32, n_dp=4, chunk_size=200,
    n_mamba_dp=2, d_state=16, mamba_expand=2, mamba_conv=4,
).cuda()
out = model(torch.randn(1, 48000).cuda())
print("10. full model output:", out.shape, flush=True)
out.sum().backward()
print("11. full backward OK", flush=True)
print("ALL DONE", flush=True)
PYEOF
echo "exit code: $?"
