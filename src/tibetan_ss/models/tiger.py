"""Adapter for TIGER (Xu et al., ICLR 2025).

Upstream: https://github.com/JusperLee/TIGER

The native ``TIGER`` module returns ``(B, num_sources, T)`` directly, so the
adapter is a thin wrapper that only canonicalises the input shape and sample
rate, and exposes a ``build_tiger`` factory for the registry.
"""

from __future__ import annotations

from typing import Any

import torch

from ._thirdparty_path import register_thirdparty
from .base import BaseSeparator
from .registry import register


class TIGERAdapter(BaseSeparator):
    """Wraps ``look2hear.models.TIGER`` as a ``BaseSeparator``."""

    def __init__(self, sample_rate: int = 16000, num_speakers: int = 2, **kwargs: Any):
        super().__init__(num_speakers=num_speakers, sample_rate=sample_rate)
        register_thirdparty("TIGER")
        from look2hear.models.tiger import TIGER as _TIGER
        # Default hyper-params match TIGER-S (efficient) configuration on
        # 16 kHz speech separation; see configs/speech-32k.yaml in the upstream repo.
        hparams = {
            "out_channels":   kwargs.get("out_channels",   128),
            "in_channels":    kwargs.get("in_channels",    128),
            "num_blocks":     kwargs.get("num_blocks",     16),
            "upsampling_depth": kwargs.get("upsampling_depth", 4),
            "att_n_head":     kwargs.get("att_n_head",     4),
            "att_hid_chan":   kwargs.get("att_hid_chan",   4),
            "att_kernel_size": kwargs.get("att_kernel_size", 8),
            "att_stride":     kwargs.get("att_stride",     1),
            "win":            kwargs.get("win",            1024),
            "stride":         kwargs.get("stride",         256),
            "num_sources":    num_speakers,
            "sample_rate":    sample_rate,
        }
        self.model = _TIGER(**hparams)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        mix = self._prepare_input(mixture)              # (B, T)
        est = self.model(mix)                            # upstream returns (B, K, T)
        if est.ndim == 3 and est.shape[1] != self.num_speakers and est.shape[-1] == self.num_speakers:
            est = est.transpose(1, 2).contiguous()
        return est


@register("tiger")
def build_tiger(sample_rate: int = 16000, num_speakers: int = 2, **kwargs) -> TIGERAdapter:
    return TIGERAdapter(sample_rate=sample_rate, num_speakers=num_speakers, **kwargs)
