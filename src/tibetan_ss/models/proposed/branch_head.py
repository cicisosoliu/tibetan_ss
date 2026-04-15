"""Dual-branch TCN heads — the *actual* early-stage separator."""

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


class DualBranchHeads(nn.Module):
    """Two independent branch heads operating on a shared encoded feature."""

    def __init__(self, bottleneck: int = 128, hidden: int = 512,
                 tcn_blocks: int = 8, tcn_repeats: int = 4,
                 perturbation_std: float = 1e-3):
        super().__init__()
        self.branch_a = BranchHead(bottleneck, hidden, tcn_blocks, tcn_repeats)
        self.branch_b = BranchHead(bottleneck, hidden, tcn_blocks, tcn_repeats)
        self.perturbation_std = float(perturbation_std)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject a tiny per-branch perturbation (only at training) to break
        symmetry, then run each branch independently.
        """
        if self.training and self.perturbation_std > 0:
            eps_a = torch.randn_like(features) * self.perturbation_std
            eps_b = torch.randn_like(features) * self.perturbation_std
            x_a = features + eps_a
            x_b = features + eps_b
        else:
            x_a = features
            x_b = features
        z_a = self.branch_a(x_a)
        z_b = self.branch_b(x_b)
        return z_a, z_b
