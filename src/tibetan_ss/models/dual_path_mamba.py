"""Adapter for Dual-path Mamba (Jiang et al., ICASSP 2025).

Upstream: https://github.com/xi-j/Mamba-TasNet

**Why we don't use the upstream's vendored BiMamba/selective_scan:**

Mamba-TasNet ships its own ``modules/mamba/bimamba.py`` and
``selective_scan_interface.py``, which call ``causal_conv1d_cuda`` with a
5-argument signature from causal-conv1d ~1.1. Modern causal-conv1d (>=1.4)
changed the C++ binding to 8 arguments, causing a hard ``TypeError`` at
forward time. Pinning an older causal-conv1d breaks mamba-ssm >=2.

The solution here is to **bypass the vendored code entirely**: we build a
lightweight ``NativeBiMamba`` on top of ``mamba_ssm.Mamba`` (which the user
already verified works) and wrap it in a SpeechBrain ``Dual_Path_Model`` for
the dual-path framework.

.. code-block:: bash

    pip install speechbrain mamba-ssm causal-conv1d
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import BaseSeparator
from .registry import register


# ---------------------------------------------------------------------------
# Native bidirectional Mamba — uses only `mamba_ssm.Mamba` (no vendored code)
# ---------------------------------------------------------------------------

class NativeBiMamba(nn.Module):
    """Bidirectional Mamba built from two standard ``mamba_ssm.Mamba`` passes.

    Forward pass: run Mamba on ``x`` and on ``flip(x)``, concatenate along the
    feature dimension, then project back to ``d_model``.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, **kwargs):
        super().__init__()
        from mamba_ssm import Mamba
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv,
                               expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv,
                               expand=expand)
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: (B, L, D)
        fwd = self.mamba_fwd(x)
        bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        return self.out_proj(torch.cat([fwd, bwd], dim=-1))


class NativeMambaBlock(nn.Module):
    """Pre-norm + Mamba (or BiMamba) + residual."""

    def __init__(self, d_model: int, bidirectional: bool = True,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if bidirectional:
            self.mixer = NativeBiMamba(d_model, d_state=d_state, d_conv=d_conv,
                                       expand=expand)
        else:
            from mamba_ssm import Mamba
            self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv,
                               expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class NativeMambaStack(nn.Module):
    """A stack of ``n_mamba`` Mamba blocks — drop-in replacement for the
    upstream ``MambaBlocksSequential`` but using only ``mamba_ssm.Mamba``.

    SpeechBrain's ``Dual_Path_Model`` calls ``intra_model(x)`` and
    ``inter_model(x)`` where ``x`` has shape ``(B, L, D)``. The output must
    have the same shape.
    """

    def __init__(self, n_mamba: int = 1, bidirectional: bool = True,
                 d_model: int = 256, d_state: int = 16, expand: int = 2,
                 d_conv: int = 4, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            NativeMambaBlock(d_model, bidirectional=bidirectional,
                             d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_mamba)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Build the full Encoder + Dual_Path_Model + Decoder from SpeechBrain
# ---------------------------------------------------------------------------

def _build_native(
    num_speakers: int,
    n_encoder_out: int,
    kernel_size: int,
    kernel_stride: int,
    n_dp: int,
    chunk_size: int,
    n_mamba_dp: int,
    d_state: int,
    mamba_expand: int,
    mamba_conv: int,
    bidirectional: bool,
    skip_around_intra: bool,
) -> nn.Module:
    from speechbrain.lobes.models.dual_path import (
        Dual_Path_Model,
        Decoder,
        Encoder,
    )

    encoder = Encoder(kernel_size=kernel_size, out_channels=n_encoder_out)
    intra = NativeMambaStack(
        n_mamba=max(1, n_mamba_dp // 2),
        bidirectional=bidirectional,
        d_model=n_encoder_out,
        d_state=d_state,
        expand=mamba_expand,
        d_conv=mamba_conv,
    )
    inter = NativeMambaStack(
        n_mamba=max(1, n_mamba_dp // 2),
        bidirectional=bidirectional,
        d_model=n_encoder_out,
        d_state=d_state,
        expand=mamba_expand,
        d_conv=mamba_conv,
    )
    masknet = Dual_Path_Model(
        num_spks=num_speakers,
        in_channels=n_encoder_out,
        out_channels=n_encoder_out,
        num_layers=n_dp,
        K=chunk_size,
        intra_model=intra,
        inter_model=inter,
        norm="ln",
        linear_layer_after_inter_intra=False,
        skip_around_intra=skip_around_intra,
    )
    decoder = Decoder(
        in_channels=n_encoder_out,
        out_channels=1,
        kernel_size=kernel_size,
        stride=kernel_stride,
        bias=False,
    )
    return nn.ModuleDict({"encoder": encoder, "masknet": masknet, "decoder": decoder})


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class DualPathMambaAdapter(BaseSeparator):
    """Wraps SpeechBrain Encoder + dual-path native-Mamba masknet + Decoder."""

    def __init__(self,
                 sample_rate: int = 8000,
                 num_speakers: int = 2,
                 n_encoder_out: int = 256,
                 kernel_size: int = 16,
                 n_dp: int = 16,
                 chunk_size: int = 250,
                 n_mamba_dp: int = 2,
                 d_state: int = 16,
                 mamba_expand: int = 2,
                 mamba_conv: int = 4,
                 bidirectional: bool = True,
                 skip_around_intra: bool = True,
                 **_: Any):
        super().__init__(num_speakers=num_speakers, sample_rate=sample_rate)
        self.num_speakers = num_speakers
        kernel_stride = kernel_size // 2
        modules = _build_native(
            num_speakers=num_speakers,
            n_encoder_out=n_encoder_out,
            kernel_size=kernel_size,
            kernel_stride=kernel_stride,
            n_dp=n_dp,
            chunk_size=chunk_size,
            n_mamba_dp=n_mamba_dp,
            d_state=d_state,
            mamba_expand=mamba_expand,
            mamba_conv=mamba_conv,
            bidirectional=bidirectional,
            skip_around_intra=skip_around_intra,
        )
        self.encoder = modules["encoder"]
        self.masknet = modules["masknet"]
        self.decoder = modules["decoder"]
        self.n_encoder_out = n_encoder_out
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        mix = self._prepare_input(mixture)                 # (B, T)
        mix_w = self.encoder(mix)                          # (B, F, L)
        est_mask = self.masknet(mix_w)                     # (num_spks, B, F, L)
        mix_w = torch.stack([mix_w] * self.num_speakers)   # (num_spks, B, F, L)
        sep_h = mix_w * est_mask
        est = torch.stack(
            [self.decoder(sep_h[i]) for i in range(self.num_speakers)],
            dim=1,
        )                                                   # (B, num_spks, T')
        T = mix.shape[-1]
        if est.shape[-1] > T:
            est = est[..., :T]
        elif est.shape[-1] < T:
            est = torch.nn.functional.pad(est, (0, T - est.shape[-1]))
        return est


@register("dual_path_mamba")
def build_dual_path_mamba(sample_rate: int = 8000, num_speakers: int = 2, **kwargs):
    return DualPathMambaAdapter(sample_rate=sample_rate, num_speakers=num_speakers, **kwargs)
