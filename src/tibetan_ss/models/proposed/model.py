"""The proposed model: Early-Separation dual-branch TCN + shared decoder.

Per ``提出模型.docx``, the separation starts *at the encoder stage*:

1. The input mixture is duplicated into two copies with tiny perturbation.
2. **Both copies pass through the shared encoder independently** — the
   shared weights + different inputs cause the encoded representations to
   diverge from the very first layer.
3. Each encoded representation then goes through its own independent branch
   head to further reinforce speaker-specific patterns.
4. A shared mask-based decoder reconstructs per-speaker waveforms.

This is the "distributed separation" design: Encoder begins separation →
Branch reinforces → Decoder completes → Discriminator purifies.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from ..base import BaseSeparator
from ..registry import register
from .branch_head import BranchHead
from .decoder import SharedMaskDecoder, WaveformDecoder
from .encoder import SharedEncoder


@dataclass
class ProposedConfig:
    n_filters: int = 512
    kernel_size: int = 32
    stride: int | None = None
    bottleneck: int = 128
    tcn_hidden: int = 512
    encoder_tcn_blocks: int = 8
    encoder_tcn_repeats: int = 3
    branch_tcn_blocks: int = 8
    branch_tcn_repeats: int = 4
    decoder_tcn_blocks: int = 8
    decoder_tcn_repeats: int = 3
    mask_nonlinear: str = "relu"
    perturbation_std: float = 1e-3


class ProposedEarlySeparation(BaseSeparator):
    """Early-Separation dual-branch network.

    Forward pass (matches docx §3)::

        mix_a = mix + ε_a                              # §3.1 双分支输入构建
        mix_b = mix + ε_b
        encoded_a, features_a = encoder(mix_a)         # §3.2 共享编码器（跑两次）
        encoded_b, features_b = encoder(mix_b)
        z_a = branch_a(features_a)                     # §3.3 双分支表示头
        z_b = branch_b(features_b)
        sep_a = mask_decoder(z_a, encoded_a)           # §3.5 共享解码器
        sep_b = mask_decoder(z_b, encoded_b)
        wav_a = wave_decoder(sep_a)
        wav_b = wave_decoder(sep_b)

    The encoder is shared (same weights) but receives **different inputs**,
    so the two branches diverge from the very first layer — this is the
    "Early Separation" core idea.
    """

    def __init__(self, sample_rate: int = 16000, num_speakers: int = 2,
                 config: ProposedConfig | dict | None = None):
        super().__init__(num_speakers=num_speakers, sample_rate=sample_rate)
        if isinstance(config, dict):
            allowed = {f.name for f in fields(ProposedConfig)}
            config = ProposedConfig(**{k: v for k, v in config.items() if k in allowed})
        cfg = config or ProposedConfig()
        self.cfg = cfg
        self.perturbation_std = float(cfg.perturbation_std)

        # Shared encoder — called twice (once per branch) with different inputs
        self.encoder = SharedEncoder(
            n_filters=cfg.n_filters, kernel_size=cfg.kernel_size, stride=cfg.stride,
            bottleneck=cfg.bottleneck, hidden=cfg.tcn_hidden,
            tcn_blocks=cfg.encoder_tcn_blocks, tcn_repeats=cfg.encoder_tcn_repeats,
        )
        # Two independent branch heads (no perturbation here — it's done at input level)
        self.branch_a = BranchHead(
            bottleneck=cfg.bottleneck, hidden=cfg.tcn_hidden,
            tcn_blocks=cfg.branch_tcn_blocks, tcn_repeats=cfg.branch_tcn_repeats,
        )
        self.branch_b = BranchHead(
            bottleneck=cfg.bottleneck, hidden=cfg.tcn_hidden,
            tcn_blocks=cfg.branch_tcn_blocks, tcn_repeats=cfg.branch_tcn_repeats,
        )
        self.mask_decoder = SharedMaskDecoder(
            bottleneck=cfg.bottleneck, n_filters=cfg.n_filters, hidden=cfg.tcn_hidden,
            tcn_blocks=cfg.decoder_tcn_blocks, tcn_repeats=cfg.decoder_tcn_repeats,
            mask_nonlinear=cfg.mask_nonlinear,
        )
        self.wave_decoder = WaveformDecoder(
            n_filters=cfg.n_filters, kernel_size=cfg.kernel_size, stride=cfg.stride,
        )

    # ------------------------------------------------------------------
    def forward(self, mixture: torch.Tensor, return_aux: bool = False):
        mix = self._prepare_input(mixture)                              # (B, T)
        T = mix.shape[-1]

        # §3.1 — Dual-branch input: duplicate with tiny perturbation
        if self.training and self.perturbation_std > 0:
            eps_a = torch.randn_like(mix) * self.perturbation_std
            eps_b = torch.randn_like(mix) * self.perturbation_std
            mix_a = mix + eps_a
            mix_b = mix + eps_b
        else:
            mix_a = mix
            mix_b = mix

        # §3.2 — Shared encoder runs TWICE on different inputs
        encoded_a, features_a = self.encoder(mix_a)
        encoded_b, features_b = self.encoder(mix_b)

        # §3.3 — Independent branch heads
        z_a = self.branch_a(features_a)
        z_b = self.branch_b(features_b)

        # §3.5 — Shared mask-based decoder
        sep_a = self.mask_decoder(z_a, encoded_a)
        sep_b = self.mask_decoder(z_b, encoded_b)
        wav_a = self.wave_decoder(sep_a, output_length=T)
        wav_b = self.wave_decoder(sep_b, output_length=T)

        est = torch.stack([wav_a, wav_b], dim=1)                        # (B, 2, T)
        if return_aux:
            return est, {
                "z_a": z_a, "z_b": z_b,
                "encoded_a": encoded_a, "encoded_b": encoded_b,
                "features_a": features_a, "features_b": features_b,
            }
        return est


@register("proposed")
def build_proposed(sample_rate: int = 16000, num_speakers: int = 2, **cfg) -> ProposedEarlySeparation:
    return ProposedEarlySeparation(sample_rate=sample_rate, num_speakers=num_speakers, config=cfg)
