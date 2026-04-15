"""Base separator interface.

All models in this repo must inherit ``BaseSeparator`` and conform to the
signature::

    def forward(self, mixture: Tensor[(B, T) | (B, 1, T)]) -> Tensor[(B, C, T)]

The unified trainer / evaluator relies on this contract — never return a
list/tuple of per-speaker tensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaseSeparator(nn.Module):
    """Common base class for every separator in this project.

    Attributes
    ----------
    num_speakers : int
        Number of output sources (fixed to 2 for this project).
    sample_rate : int
        Sampling rate the model assumes. Used by evaluation utilities.
    """

    num_speakers: int = 2
    sample_rate: int = 16000

    def __init__(self, num_speakers: int = 2, sample_rate: int = 16000) -> None:
        super().__init__()
        self.num_speakers = int(num_speakers)
        self.sample_rate = int(sample_rate)

    # ------------------------------------------------------------------
    def _prepare_input(self, mixture: torch.Tensor) -> torch.Tensor:
        """Canonicalise the input to shape ``(B, T)``."""
        if mixture.ndim == 3:
            if mixture.shape[1] != 1:
                raise ValueError(
                    f"Expected mono mixture (B, 1, T) – got {tuple(mixture.shape)}"
                )
            mixture = mixture.squeeze(1)
        elif mixture.ndim != 2:
            raise ValueError(f"Expected (B, T) input – got {tuple(mixture.shape)}")
        return mixture

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class Identity2Speaker(BaseSeparator):
    """Trivial placeholder that returns the mixture as every source.

    Used by unit tests and as a smoke-test model when the real heavyweight
    repos aren't installed yet.
    """

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        mix = self._prepare_input(mixture)
        return mix.unsqueeze(1).expand(-1, self.num_speakers, -1).contiguous()
