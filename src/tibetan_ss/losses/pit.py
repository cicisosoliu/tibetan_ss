"""Permutation-Invariant Training (uPIT) for 2-speaker separation.

For the 2-speaker case there are only two possible permutations, so we
enumerate them explicitly (O(C!) with C=2 ⇒ 2 evaluations) rather than
invoking the Hungarian algorithm. This is numerically identical to
Asteroid's ``PITLossWrapper`` and an order of magnitude faster on batch.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .sisdr import si_sdr

# ---------------------------------------------------------------------------
# Generic N-speaker PIT (O(C!)): we only actually use C=2 here, but the
# implementation supports arbitrary C for future extension.
# ---------------------------------------------------------------------------

from itertools import permutations


def _pairwise_loss(est: torch.Tensor, ref: torch.Tensor,
                   pair_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
                   ) -> torch.Tensor:
    """Compute ``pair_fn`` for every (est_i, ref_j) pair.

    Returns tensor of shape ``(B, C, C)`` where entry ``(b, i, j)`` is the
    per-utterance score between ``est[b, i]`` and ``ref[b, j]``.
    """
    B, C, T = est.shape
    est_e = est.unsqueeze(2).expand(B, C, C, T)          # (B, C_est, C_ref, T)
    ref_e = ref.unsqueeze(1).expand(B, C, C, T)
    return pair_fn(est_e.reshape(-1, T), ref_e.reshape(-1, T)).reshape(B, C, C)


def pit_si_sdr_loss(est: torch.Tensor, ref: torch.Tensor,
                    return_perm: bool = False
                    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """PIT-wrapped neg-SI-SDR loss.

    Parameters
    ----------
    est : (B, C, T) estimated sources
    ref : (B, C, T) reference sources
    return_perm : whether to return the best permutation per batch entry.

    Returns
    -------
    loss  : scalar tensor (mean neg-SI-SDR over the best permutation)
    perm  : (B, C) long tensor if ``return_perm``
    """
    if est.shape != ref.shape:
        raise ValueError(f"Shape mismatch: est={tuple(est.shape)}, ref={tuple(ref.shape)}")
    B, C, _ = est.shape
    pair_scores = _pairwise_loss(est, ref, si_sdr)             # (B, C, C) – higher is better

    perms = list(permutations(range(C)))                        # C! permutations
    perm_tensor = torch.tensor(perms, device=est.device, dtype=torch.long)  # (P, C)
    idx0 = torch.arange(C, device=est.device)
    P = perm_tensor.shape[0]
    scores = torch.empty(B, P, device=est.device, dtype=est.dtype)
    for i, perm in enumerate(perm_tensor):
        scores[:, i] = pair_scores[:, idx0, perm].mean(dim=-1)

    best = scores.max(dim=1)
    loss = -best.values.mean()
    if return_perm:
        return loss, perm_tensor[best.indices]
    return loss


class PITWrapper(nn.Module):
    """Lightweight wrapper around :func:`pit_si_sdr_loss`."""

    def __init__(self, return_perm: bool = False):
        super().__init__()
        self.return_perm = return_perm

    def forward(self, est: torch.Tensor, ref: torch.Tensor):
        return pit_si_sdr_loss(est, ref, return_perm=self.return_perm)


def reorder_sources(est: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Given best permutation ``perm`` from :func:`pit_si_sdr_loss`, reorder
    ``est`` so that ``est[b, i]`` aligns with ``ref[b, i]``.
    """
    B, C, T = est.shape
    batch_idx = torch.arange(B, device=est.device).unsqueeze(-1).expand(B, C)
    return est[batch_idx, perm]
