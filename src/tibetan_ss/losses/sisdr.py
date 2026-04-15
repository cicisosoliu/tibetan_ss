"""Scale-Invariant SDR loss (as in Conv-TasNet and Le Roux et al. 2019)."""

from __future__ import annotations

import torch
import torch.nn as nn

_EPS = 1e-8


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """Return the SI-SDR in dB for ``est`` w.r.t. ``ref`` (shape ``(..., T)``).

    The inputs are first zero-mean normalised along the time dimension.
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    dot = torch.sum(est * ref, dim=-1, keepdim=True)
    denom = torch.sum(ref * ref, dim=-1, keepdim=True) + eps
    s_target = dot / denom * ref
    e_noise = est - s_target
    return 10 * torch.log10(
        (torch.sum(s_target * s_target, dim=-1) + eps)
        / (torch.sum(e_noise * e_noise, dim=-1) + eps)
    )


def neg_si_sdr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Mean negative SI-SDR — ready to plug into ``loss.backward()``."""
    return -si_sdr(est, ref).mean()


class SISDRLoss(nn.Module):
    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return neg_si_sdr(est, ref)
