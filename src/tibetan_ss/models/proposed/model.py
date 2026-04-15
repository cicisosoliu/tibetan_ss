"""The proposed model: Early-Separation dual-branch TCN + shared decoder.

Wires the pieces together so the trainer can call::

    est = model(mixture)                       # (B, 2, T)
    est, aux = model(mixture, return_aux=True) # + {'z_a', 'z_b', 'encoded'}

``aux`` is used by the GAN trainer to compute L_rep and feed the discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..base import BaseSeparator
from ..registry import register
from .branch_head import DualBranchHeads
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

    Forward pass::

        encoded, features = encoder(mixture)
        z_a, z_b          = branch_heads(features)
        sep_a             = decoder(z_a, encoded)          # feature basis
        sep_b             = decoder(z_b, encoded)
        wav_a             = wave_decoder(sep_a)
        wav_b             = wave_decoder(sep_b)
    """

    def __init__(self, sample_rate: int = 16000, num_speakers: int = 2,
                 config: ProposedConfig | dict | None = None):
        super().__init__(num_speakers=num_speakers, sample_rate=sample_rate)
        if isinstance(config, dict):
            config = ProposedConfig(**config)
        cfg = config or ProposedConfig()
        self.cfg = cfg
        self.encoder = SharedEncoder(
            n_filters=cfg.n_filters, kernel_size=cfg.kernel_size, stride=cfg.stride,
            bottleneck=cfg.bottleneck, hidden=cfg.tcn_hidden,
            tcn_blocks=cfg.encoder_tcn_blocks, tcn_repeats=cfg.encoder_tcn_repeats,
        )
        self.branches = DualBranchHeads(
            bottleneck=cfg.bottleneck, hidden=cfg.tcn_hidden,
            tcn_blocks=cfg.branch_tcn_blocks, tcn_repeats=cfg.branch_tcn_repeats,
            perturbation_std=cfg.perturbation_std,
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
        mix = self._prepare_input(mixture)
        T = mix.shape[-1]
        encoded, features = self.encoder(mix)
        z_a, z_b = self.branches(features)
        sep_a = self.mask_decoder(z_a, encoded)
        sep_b = self.mask_decoder(z_b, encoded)
        wav_a = self.wave_decoder(sep_a, output_length=T)
        wav_b = self.wave_decoder(sep_b, output_length=T)
        est = torch.stack([wav_a, wav_b], dim=1)          # (B, 2, T)
        if return_aux:
            return est, {"z_a": z_a, "z_b": z_b, "encoded": encoded, "features": features}
        return est


@register("proposed")
def build_proposed(sample_rate: int = 16000, num_speakers: int = 2, **cfg) -> ProposedEarlySeparation:
    return ProposedEarlySeparation(sample_rate=sample_rate, num_speakers=num_speakers, config=cfg)
