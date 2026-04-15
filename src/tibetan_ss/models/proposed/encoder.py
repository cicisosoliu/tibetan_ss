"""Shared encoder of the proposed model: Conv encoder + TCN × 3."""

from __future__ import annotations

import torch
import torch.nn as nn

from .tcn import GlobalLayerNorm, TCNStack


class WaveformEncoder(nn.Module):
    """1-D conv encoder (Conv-TasNet style).

    Parameters
    ----------
    n_filters : int
        Number of output channels (``N``).
    kernel_size : int
        Filter length in *samples*. A common choice is 16 @ 8 kHz / 32 @ 16 kHz.
    stride : int
        Typically ``kernel_size // 2``.
    """

    def __init__(self, n_filters: int = 512, kernel_size: int = 32, stride: int | None = None):
        super().__init__()
        stride = stride or kernel_size // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv1d(1, n_filters, kernel_size, stride=stride, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)                                           # (B, 1, T)
        return self.act(self.conv(x))                                     # (B, N, L)


class SharedEncoder(nn.Module):
    """Shared "Conv + TCN × 3" encoder.

    Accepts a mixture waveform and returns an encoded feature map
    ``E ∈ R^{B × C × L}`` where ``C = bottleneck`` and ``L`` is the downsampled
    sequence length.
    """

    def __init__(self,
                 n_filters: int = 512,
                 kernel_size: int = 32,
                 stride: int | None = None,
                 bottleneck: int = 128,
                 hidden: int = 512,
                 tcn_blocks: int = 8,
                 tcn_repeats: int = 3):
        super().__init__()
        self.wave_encoder = WaveformEncoder(n_filters, kernel_size, stride)
        self.norm = GlobalLayerNorm(n_filters)
        self.reduce = nn.Conv1d(n_filters, bottleneck, 1)
        # "TCN × 3" in the spec — interpreted as 3 repeats of an 8-block dilated stack
        self.tcn = TCNStack(channels=bottleneck, hidden=hidden,
                            kernel=3, num_blocks=tcn_blocks, repeats=tcn_repeats)
        self.n_filters = n_filters
        self.bottleneck = bottleneck

    def forward(self, mixture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns
        -------
        encoded : (B, N, L) raw encoder filter bank output (for mask re-application)
        features : (B, bottleneck, L) bottleneck features fed to the branch heads
        """
        encoded = self.wave_encoder(mixture)                               # (B, N, L)
        features = self.reduce(self.norm(encoded))                         # (B, C, L)
        features = self.tcn(features)                                      # (B, C, L)
        return encoded, features
