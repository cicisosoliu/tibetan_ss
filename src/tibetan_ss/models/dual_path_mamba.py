"""Adapter for Dual-path Mamba (Jiang et al., ICASSP 2025).

Upstream: https://github.com/xi-j/Mamba-TasNet

The upstream code relies on SpeechBrain's ``Encoder``/``Decoder`` wrappers
plus ``speechbrain.lobes.models.dual_path.Dual_Path_Model``. We re-use those
pieces directly here and only add a thin nn.Module that packages them into
``(B, T) -> (B, 2, T)``. SpeechBrain and mamba-ssm must be installed:

.. code-block:: bash

    pip install speechbrain mamba-ssm causal-conv1d
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._thirdparty_path import register_thirdparty
from .base import BaseSeparator
from .registry import register


def _build_native(
    sample_rate: int,
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
    register_thirdparty("Mamba-TasNet")

    # ---- Fix: mamba-ssm >= 2.0 moved RMSNorm; upstream repo still tries the
    # old path, gets None, then partial(None, ...) → TypeError.  We patch it
    # before anything else imports the module. ---
    import modules.mamba_blocks as _mb
    if _mb.RMSNorm is None:
        try:
            from mamba_ssm.ops.triton.layer_norm import RMSNorm
            _mb.RMSNorm = RMSNorm
        except ImportError:
            class _RMSNorm(nn.Module):
                def __init__(self, hidden_size, eps=1e-6, **kwargs):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(hidden_size))
                    self.eps = eps
                def forward(self, x):
                    x_f = x.float()
                    return (self.weight * x_f * torch.rsqrt(
                        x_f.pow(2).mean(-1, keepdim=True) + self.eps
                    )).to(x.dtype)
            _mb.RMSNorm = _RMSNorm

    from speechbrain.lobes.models.dual_path import (
        Dual_Path_Model,
        Decoder,
        Encoder,
    )
    from modules.mamba_blocks import MambaBlocksSequential

    encoder = Encoder(kernel_size=kernel_size, out_channels=n_encoder_out)
    intra = MambaBlocksSequential(
        n_mamba=n_mamba_dp // 2,
        bidirectional=bidirectional,
        d_model=n_encoder_out,
        d_state=d_state,
        expand=mamba_expand,
        d_conv=mamba_conv,
        fused_add_norm=False,
        rms_norm=True,
        residual_in_fp32=False,
        conv_bias=True,
        bias=False,
    )
    inter = MambaBlocksSequential(
        n_mamba=n_mamba_dp // 2,
        bidirectional=bidirectional,
        d_model=n_encoder_out,
        d_state=d_state,
        expand=mamba_expand,
        d_conv=mamba_conv,
        fused_add_norm=False,
        rms_norm=True,
        residual_in_fp32=False,
        conv_bias=True,
        bias=False,
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


class DualPathMambaAdapter(BaseSeparator):
    """Wraps Mamba-TasNet's encoder + dual-path Mamba masknet + decoder."""

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
            sample_rate=sample_rate,
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
        # SpeechBrain's Encoder takes (B, T) and returns (B, F, L)
        mix_w = self.encoder(mix)
        est_mask = self.masknet(mix_w)                      # (num_spks, B, F, L)
        mix_w = torch.stack([mix_w] * self.num_speakers)    # (num_spks, B, F, L)
        sep_h = mix_w * est_mask
        # Decode each speaker
        est = torch.stack(
            [self.decoder(sep_h[i]) for i in range(self.num_speakers)],
            dim=1,
        )                                                    # (B, num_spks, T')
        T = mix.shape[-1]
        if est.shape[-1] > T:
            est = est[..., :T]
        elif est.shape[-1] < T:
            est = torch.nn.functional.pad(est, (0, T - est.shape[-1]))
        return est


@register("dual_path_mamba")
def build_dual_path_mamba(sample_rate: int = 8000, num_speakers: int = 2, **kwargs):
    return DualPathMambaAdapter(sample_rate=sample_rate, num_speakers=num_speakers, **kwargs)
