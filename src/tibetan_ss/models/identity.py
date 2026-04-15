"""Identity separator — smoke-test model."""

from __future__ import annotations

from .base import Identity2Speaker
from .registry import register


@register("identity")
def build_identity(num_speakers: int = 2, sample_rate: int = 16000, **kwargs) -> Identity2Speaker:
    return Identity2Speaker(num_speakers=num_speakers, sample_rate=sample_rate)
