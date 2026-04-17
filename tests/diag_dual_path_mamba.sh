#!/usr/bin/env bash
# Diagnostic: step-by-step Dual-path Mamba construction to find where it hangs.
# Usage (on GPU server):
#   cd /hy-tmp/tibetan_ss
#   bash tests/diag_dual_path_mamba.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

timeout 120 python3 << 'PYEOF'
import torch, sys, os
sys.path.insert(0, os.path.join(os.environ.get("PYTHONPATH","").split(":")[0], "..", "third_party", "Mamba-TasNet"))
# also try repo-root based path
repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in dir() else "."
sys.path.insert(0, os.path.join(repo, "third_party", "Mamba-TasNet"))

print("1. importing speechbrain...", flush=True)
from speechbrain.lobes.models.dual_path import Dual_Path_Model, Decoder, Encoder
print("2. speechbrain OK", flush=True)

print("3. patching RMSNorm...", flush=True)
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
                return (self.weight * xf * torch.rsqrt(
                    xf.pow(2).mean(-1, keepdim=True) + self.eps
                )).to(x.dtype)
        _mb.RMSNorm = _RMSNorm
print("4. RMSNorm:", _mb.RMSNorm, flush=True)

from modules.mamba_blocks import MambaBlocksSequential
print("5. building encoder...", flush=True)
encoder = Encoder(kernel_size=32, out_channels=256)
print("6. encoder OK", flush=True)

print("7. building MambaBlocksSequential (intra)...", flush=True)
intra = MambaBlocksSequential(
    n_mamba=1, bidirectional=True, d_model=256, d_state=16,
    expand=2, d_conv=4, fused_add_norm=False, rms_norm=True,
    residual_in_fp32=False, conv_bias=True, bias=False,
)
print("8. intra OK", flush=True)

print("9. building MambaBlocksSequential (inter)...", flush=True)
inter = MambaBlocksSequential(
    n_mamba=1, bidirectional=True, d_model=256, d_state=16,
    expand=2, d_conv=4, fused_add_norm=False, rms_norm=True,
    residual_in_fp32=False, conv_bias=True, bias=False,
)
print("10. inter OK", flush=True)

print("11. building Dual_Path_Model (num_layers=4, K=400)...", flush=True)
masknet = Dual_Path_Model(
    num_spks=2, in_channels=256, out_channels=256,
    num_layers=4, K=400,
    intra_model=intra, inter_model=inter,
    norm="ln", linear_layer_after_inter_intra=False,
    skip_around_intra=True,
)
print("12. masknet OK", flush=True)

print("13. building decoder...", flush=True)
decoder = Decoder(in_channels=256, out_channels=1, kernel_size=32, stride=16, bias=False)
print("14. decoder OK", flush=True)

print("15. moving to cuda...", flush=True)
encoder = encoder.cuda()
masknet = masknet.cuda()
decoder = decoder.cuda()
print("16. cuda OK", flush=True)

print("17. forward: encoder...", flush=True)
x = torch.randn(1, 48000).cuda()
mix_w = encoder(x)
print("18. encoder out:", mix_w.shape, flush=True)

print("19. forward: masknet...", flush=True)
est_mask = masknet(mix_w)
print("20. masknet out: type=", type(est_mask), flush=True)

print("ALL DONE", flush=True)
PYEOF

echo "exit code: $?"
