"""Auxiliary losses for the proposed model: L_rep, L_D, L_G.

``L_main`` (PIT + SI-SDR) is shared with the other baselines and lives in
``tibetan_ss.losses``. Only the proposal-specific pieces are here.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def representation_diff_loss(
    z_a: torch.Tensor, z_b: torch.Tensor,
    cosine_weight: float = 1.0, orthogonal_weight: float = 1.0,
) -> torch.Tensor:
    """Encourage the two branch representations to diverge.

    * Cosine similarity term – shrinks the channel-wise cosine similarity
      between ``z_a`` and ``z_b``.
    * Orthogonality term – shrinks ``|Z_a^T Z_b|_F`` after per-channel
      L2-normalisation, pushing the two subspaces apart.
    """
    # Flatten time, keep batch & channel dims
    Za = z_a.flatten(2)                                 # (B, C, L)
    Zb = z_b.flatten(2)
    # Cosine similarity along time per channel, averaged
    cos = F.cosine_similarity(Za, Zb, dim=-1).abs().mean()
    # Orthogonality: normalised Gram cross term
    za_n = F.normalize(Za, dim=-1)
    zb_n = F.normalize(Zb, dim=-1)
    cross = torch.matmul(za_n, zb_n.transpose(-1, -2))  # (B, C, C)
    # Subtract identity so the diagonal doesn't penalise same-channel corr
    eye = torch.eye(cross.shape[-1], device=cross.device, dtype=cross.dtype).expand_as(cross)
    ortho = (cross - eye).pow(2).mean()
    return cosine_weight * cos + orthogonal_weight * ortho


def hinge_discriminator_loss(d_real: Iterable[torch.Tensor],
                             d_fake: Iterable[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for the discriminator (sum over all scales)."""
    loss = 0.0
    for r, f in zip(d_real, d_fake):
        loss = loss + F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean()
    return loss


def hinge_generator_loss(d_fake: Iterable[torch.Tensor]) -> torch.Tensor:
    """Hinge loss for the generator (separator) side."""
    loss = 0.0
    for f in d_fake:
        loss = loss + (-f).mean()
    return loss


def feature_matching_loss(
    feats_real: Iterable[Iterable[torch.Tensor]],
    feats_fake: Iterable[Iterable[torch.Tensor]],
) -> torch.Tensor:
    """Optional L1 feature-matching loss. Not enabled by default; here for
    future ablation. Each inner iterable corresponds to one scale.
    """
    total = 0.0
    n = 0
    for scale_r, scale_f in zip(feats_real, feats_fake):
        for r, f in zip(scale_r, scale_f):
            total = total + F.l1_loss(f, r.detach())
            n += 1
    return total / max(n, 1)
