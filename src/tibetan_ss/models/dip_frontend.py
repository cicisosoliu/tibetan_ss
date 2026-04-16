"""DIP Frontend (Wang et al., IEEE/ACM TASLP 2024).

Only the dataset generation code is officially released (JorisCos/LibriMix),
so this file re-implements the *idea* of the paper:

* **Frontend** – a Siamese self-supervised encoder that maps a mixture to a
  domain-invariant latent representation. Implementation here is a small
  TCN stack with ``BYOL``-style self-distillation heads used only during
  frontend pre-training. During supervised separation, only the TCN stack
  is active and its weights are *frozen* by default.

* **Downstream separator** – any of our baselines. By default we pair the
  DIP frontend with a Conv-TasNet-style masknet + waveform decoder, matching
  the ablation reported in the paper.

The resulting module satisfies our ``BaseSeparator`` contract so it can be
plugged into the standard training loop.

Pre-training is *optional*: if the user has not yet pre-trained the frontend,
the model still works end-to-end (frontend is learned jointly with the
separator). Toggle via ``freeze_frontend: bool`` in the config.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from .base import BaseSeparator
from .proposed.decoder import SharedMaskDecoder, WaveformDecoder
from .proposed.encoder import SharedEncoder
from .registry import register


@dataclass
class DIPConfig:
    n_filters: int = 512
    kernel_size: int = 32
    stride: int | None = None
    bottleneck: int = 128
    tcn_hidden: int = 512
    frontend_tcn_blocks: int = 8
    frontend_tcn_repeats: int = 3
    sep_tcn_blocks: int = 8
    sep_tcn_repeats: int = 3
    mask_nonlinear: str = "relu"
    freeze_frontend: bool = False          # set True after SSL pre-training
    projection_dim: int = 256              # SSL projection head (pre-training only)


class _ProjectionHead(nn.Module):
    """Small MLP used only during SSL pre-training."""

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → (B, out_dim) via mean-pool
        return self.net(x.mean(dim=-1))


class DIPSeparator(BaseSeparator):
    """Domain-invariant pretrained frontend + Conv-TasNet separator.

    Forward returns ``(B, num_spks, T)``. ``return_features`` can be set to
    ``True`` to also return the frontend representation — useful for SSL
    pre-training loops.
    """

    def __init__(self, sample_rate: int = 16000, num_speakers: int = 2,
                 config: DIPConfig | dict | None = None):
        super().__init__(num_speakers=num_speakers, sample_rate=sample_rate)
        if isinstance(config, dict):
            allowed = {f.name for f in fields(DIPConfig)}
            config = DIPConfig(**{k: v for k, v in config.items() if k in allowed})
        cfg = config or DIPConfig()
        self.cfg = cfg

        self.frontend = SharedEncoder(
            n_filters=cfg.n_filters, kernel_size=cfg.kernel_size, stride=cfg.stride,
            bottleneck=cfg.bottleneck, hidden=cfg.tcn_hidden,
            tcn_blocks=cfg.frontend_tcn_blocks, tcn_repeats=cfg.frontend_tcn_repeats,
        )
        self.frontend_projection = _ProjectionHead(
            cfg.bottleneck, cfg.tcn_hidden, cfg.projection_dim,
        )
        # Momentum copy for BYOL-style SSL (pre-training only)
        self._target = None                                        # populated lazily

        # Downstream separator: a Conv-TasNet-style mask predictor per speaker
        self.separator = SharedMaskDecoder(
            bottleneck=cfg.bottleneck, n_filters=cfg.n_filters, hidden=cfg.tcn_hidden,
            tcn_blocks=cfg.sep_tcn_blocks, tcn_repeats=cfg.sep_tcn_repeats,
            mask_nonlinear=cfg.mask_nonlinear,
        )
        # One separate decoder head per speaker – lightweight 1×1 conv for speaker split
        self.speaker_split = nn.Conv1d(cfg.bottleneck, cfg.bottleneck * num_speakers, 1)
        self.wave_decoder = WaveformDecoder(
            n_filters=cfg.n_filters, kernel_size=cfg.kernel_size, stride=cfg.stride,
        )
        if cfg.freeze_frontend:
            for p in self.frontend.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------
    def target_frontend(self) -> SharedEncoder:
        """Return the momentum-EMA copy of the frontend (for BYOL SSL)."""
        if self._target is None:
            self._target = copy.deepcopy(self.frontend).eval()
            for p in self._target.parameters():
                p.requires_grad_(False)
        return self._target

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996) -> None:
        """Perform one momentum update of the target encoder (for SSL)."""
        online = self.frontend
        target = self.target_frontend()
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            p_t.data.mul_(momentum).add_(p_o.data, alpha=1 - momentum)

    # ------------------------------------------------------------------
    def forward(self, mixture: torch.Tensor, return_features: bool = False):
        mix = self._prepare_input(mixture)
        T = mix.shape[-1]
        encoded, features = self.frontend(mix)
        # Split into per-speaker representations
        per_spk = self.speaker_split(features)                              # (B, C*num_spks, L)
        per_spk = per_spk.view(per_spk.shape[0], self.num_speakers, self.cfg.bottleneck, -1)
        est_waves = []
        for k in range(self.num_speakers):
            sep_h = self.separator(per_spk[:, k], encoded)
            wav = self.wave_decoder(sep_h, output_length=T)
            est_waves.append(wav)
        est = torch.stack(est_waves, dim=1)
        if return_features:
            return est, {"features": features, "encoded": encoded}
        return est


@register("dip_frontend")
def build_dip_frontend(sample_rate: int = 16000, num_speakers: int = 2, **cfg) -> DIPSeparator:
    return DIPSeparator(sample_rate=sample_rate, num_speakers=num_speakers, config=cfg)
