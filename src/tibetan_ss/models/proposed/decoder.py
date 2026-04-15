"""Shared mask-based decoder and waveform reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn

from .tcn import TCNStack


class SharedMaskDecoder(nn.Module):
    """TCN × 3 → mask → apply to encoded mixture features.

    The decoder is *shared* between the two branches: each speaker-biased
    representation is routed through the same weights and produces its own
    mask, which is then multiplied onto the encoder filter-bank output.
    """

    def __init__(self,
                 bottleneck: int = 128,
                 n_filters: int = 512,
                 hidden: int = 512,
                 tcn_blocks: int = 8,
                 tcn_repeats: int = 3,
                 mask_nonlinear: str = "relu"):
        super().__init__()
        self.tcn = TCNStack(channels=bottleneck, hidden=hidden,
                            kernel=3, num_blocks=tcn_blocks, repeats=tcn_repeats)
        self.mask_proj = nn.Conv1d(bottleneck, n_filters, 1)
        if mask_nonlinear == "relu":
            self.mask_act = nn.ReLU()
        elif mask_nonlinear == "sigmoid":
            self.mask_act = nn.Sigmoid()
        else:
            raise ValueError(mask_nonlinear)

    def forward(self, representation: torch.Tensor, encoded_mixture: torch.Tensor) -> torch.Tensor:
        r = self.tcn(representation)
        mask = self.mask_act(self.mask_proj(r))
        return mask * encoded_mixture            # element-wise in the encoder basis


class WaveformDecoder(nn.Module):
    """Transpose-conv synthesis back to waveform (matches ``WaveformEncoder``)."""

    def __init__(self, n_filters: int = 512, kernel_size: int = 32, stride: int | None = None):
        super().__init__()
        stride = stride or kernel_size // 2
        self.deconv = nn.ConvTranspose1d(n_filters, 1, kernel_size, stride=stride, bias=False)

    def forward(self, x: torch.Tensor, output_length: int | None = None) -> torch.Tensor:
        y = self.deconv(x).squeeze(1)
        if output_length is not None:
            if y.shape[-1] > output_length:
                y = y[..., :output_length]
            elif y.shape[-1] < output_length:
                y = torch.nn.functional.pad(y, (0, output_length - y.shape[-1]))
        return y
