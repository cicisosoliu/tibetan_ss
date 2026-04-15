"""Conv-TasNet-style TCN building blocks used by the proposed model.

Implements the classical dilated depthwise-separable 1D-conv block from
``Conv-TasNet`` (Luo & Mesgarani 2019), wrapped with gLN (global layer norm).
The rest of the proposed architecture composes several stacks of these.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLayerNorm(nn.Module):
    """Global Layer Norm as in Conv-TasNet — normalises over (C, T)."""

    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class DepthwiseSeparableConv1d(nn.Module):
    """1D depthwise-separable conv with dilation."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size, padding=(kernel_size - 1) * dilation // 2,
                            dilation=dilation, groups=in_ch)
        self.pw = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class TCNBlock(nn.Module):
    """A single residual 1-D convolution block (Conv-TasNet)."""

    def __init__(self, channels: int, hidden: int, kernel: int = 3, dilation: int = 1):
        super().__init__()
        self.expand = nn.Conv1d(channels, hidden, 1)
        self.act1 = nn.PReLU(hidden)
        self.norm1 = GlobalLayerNorm(hidden)
        self.dconv = DepthwiseSeparableConv1d(hidden, hidden, kernel, dilation)
        self.act2 = nn.PReLU(hidden)
        self.norm2 = GlobalLayerNorm(hidden)
        self.shrink = nn.Conv1d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(self.act1(self.expand(x)))
        x = self.norm2(self.act2(self.dconv(x)))
        x = self.shrink(x)
        return x + residual


class TCNStack(nn.Module):
    """A stack of ``num_blocks`` TCN blocks with exponentially growing dilation.

    The stack may be repeated ``repeats`` times; each repeat resets the
    dilation to 1 and doubles it up to ``2**(num_blocks-1)``.
    """

    def __init__(self, channels: int, hidden: int = 512, kernel: int = 3,
                 num_blocks: int = 8, repeats: int = 1):
        super().__init__()
        layers = []
        for _ in range(repeats):
            for b in range(num_blocks):
                layers.append(TCNBlock(channels, hidden, kernel, dilation=2 ** b))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
