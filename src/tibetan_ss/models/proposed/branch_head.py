"""Dual-branch TCN heads — reinforce the per-speaker representations that
the shared encoder has already begun to separate.

Perturbation / symmetry-breaking is now handled at the *input waveform* level
inside ``ProposedEarlySeparation.forward`` (before the encoder), matching
``提出模型.docx §3.1``.  These branch heads are simple independent TCN stacks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .tcn import TCNStack


class BranchHead(nn.Module):
    """A single branch head (TCN × 4), independent per speaker."""

    def __init__(self, bottleneck: int = 128, hidden: int = 512,
                 tcn_blocks: int = 8, tcn_repeats: int = 4):
        super().__init__()
        self.tcn = TCNStack(channels=bottleneck, hidden=hidden,
                            kernel=3, num_blocks=tcn_blocks, repeats=tcn_repeats)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.tcn(features)
