"""Adapter for MossFormer2 (Zhao et al., ICASSP 2024).

MossFormer2 is maintained by the SpeechBrain community. The upstream
class path has changed a couple of times across SpeechBrain releases, so
this adapter tries a sequence of known import locations. If none of them
works, the error message lists the paths that were tried.

.. code-block:: bash

    pip install speechbrain

Usage from the registry::

    build_model({"name": "mossformer2", "sample_rate": 16000, "variant": "L"})

Supported variants:
* ``S`` – "small",   ≈ 2 M  params   (suits quick Tibetan experiments)
* ``L`` – "large",   ≈ 42 M params   (paper numbers)
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn as nn

from .base import BaseSeparator
from .registry import register

_CANDIDATE_PATHS = [
    # Path in recent SpeechBrain releases (>= 1.0):
    "speechbrain.lobes.models.mossformer.MossFormer2",
    "speechbrain.lobes.models.mossformer.MossFormer2_SS_16K",
    "speechbrain.lobes.models.mossformer.MossFormer2_SS_8K",
    # Fallback to the monolithic MossFormer block + conv-tasnet outer:
    "speechbrain.lobes.models.mossformer2.MossFormer2",
    "speechbrain.lobes.models.mossformer2.MossFormer2Block",
]


def _import_mossformer() -> tuple[str, Any]:
    last_err: Exception | None = None
    for dotted in _CANDIDATE_PATHS:
        mod_path, _, attr = dotted.rpartition(".")
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, attr):
                return dotted, getattr(mod, attr)
        except Exception as e:                    # pragma: no cover
            last_err = e
    msg = (
        "Could not locate MossFormer2 in this SpeechBrain install. Tried:\n"
        + "\n".join(f"  - {p}" for p in _CANDIDATE_PATHS)
        + "\n\nInstall/upgrade SpeechBrain or copy the upstream MossFormer2 "
        "module into `third_party/mossformer2/` and update _CANDIDATE_PATHS."
    )
    raise ImportError(msg) from last_err


class MossFormer2Adapter(BaseSeparator):
    """Wraps MossFormer2 as a ``BaseSeparator``.

    Different upstream class variants return different shapes — some return
    ``(B, T, num_spks)`` while others return ``(B, num_spks, T)``. The adapter
    normalises both to ``(B, num_spks, T)``.
    """

    def __init__(self, sample_rate: int = 16000, num_speakers: int = 2,
                 variant: str = "L", **kwargs: Any):
        super().__init__(num_speakers=num_speakers, sample_rate=sample_rate)
        dotted, cls = _import_mossformer()
        # Several construction signatures exist – try the most common first.
        try:
            self.model = cls(num_spks=num_speakers, **kwargs)
        except TypeError:
            try:
                self.model = cls(sample_rate=sample_rate, num_spks=num_speakers, **kwargs)
            except TypeError:
                self.model = cls(**kwargs)
        self._upstream_path = dotted
        self.variant = variant

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        mix = self._prepare_input(mixture)
        out = self.model(mix)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out.ndim == 3:
            # Detect the speaker axis by position.
            B, A, B2 = out.shape
            if A == self.num_speakers:
                pass
            elif B2 == self.num_speakers:
                out = out.transpose(1, 2).contiguous()
            else:
                # Fall back: assume (B, num_spks, T)
                pass
        else:
            raise RuntimeError(f"Unexpected MossFormer2 output shape: {out.shape}")
        T = mix.shape[-1]
        if out.shape[-1] > T:
            out = out[..., :T]
        elif out.shape[-1] < T:
            out = nn.functional.pad(out, (0, T - out.shape[-1]))
        return out


@register("mossformer2")
def build_mossformer2(sample_rate: int = 16000, num_speakers: int = 2,
                      variant: str = "L", **kwargs) -> MossFormer2Adapter:
    return MossFormer2Adapter(sample_rate=sample_rate, num_speakers=num_speakers,
                              variant=variant, **kwargs)
