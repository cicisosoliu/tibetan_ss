"""Adapter for SepReformer (Shin et al., NeurIPS 2024).

Upstream: https://github.com/dmlguq456/SepReformer

The upstream ``Model`` class returns ``(audio, audio_aux)`` where ``audio`` is
a Python list of length ``num_spks``. We stack that list, drop the auxiliary
outputs (used for deep supervision during training — can be re-enabled via
``return_aux=True``) and align the tensor with our ``(B, C, T)`` contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from ._thirdparty_path import register_thirdparty
from .base import BaseSeparator
from .registry import register

# ---------------------------------------------------------------------------
# Hard-coded config for the *Base* variant trained on WSJ0-2mix (8 kHz).
# We override sample-rate-dependent fields at construction time.
# ---------------------------------------------------------------------------


def _load_sepreformer_config(variant: str) -> dict[str, Any]:
    """Load the upstream ``configs.yaml`` for a SepReformer variant."""
    root = register_thirdparty("SepReformer")
    cfg_path = Path(root) / "models" / variant / "configs.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["config"]


class SepReformerAdapter(BaseSeparator):
    """Wraps SepReformer's ``Model`` class."""

    def __init__(self, sample_rate: int = 16000, num_speakers: int = 2,
                 variant: str = "SepReformer_Base_WSJ0", **overrides: Any):
        super().__init__(num_speakers=num_speakers, sample_rate=sample_rate)
        root = register_thirdparty("SepReformer")
        # Upstream uses relative imports anchored at its own root dir
        import sys
        import importlib
        for m in list(sys.modules):
            if m.startswith(variant.split('_')[0]) and m != __name__:
                pass
        # Ensure local utils from SepReformer are importable
        sys.path.insert(0, str(root))
        module_path = f"models.{variant}.model"
        model_mod = importlib.import_module(module_path)

        cfg = _load_sepreformer_config(variant)["model"]
        cfg["num_spks"] = num_speakers
        if overrides:
            cfg.update({k: v for k, v in overrides.items() if k in cfg})
        self.model = model_mod.Model(**cfg)
        self.variant = variant
        self._return_aux = False

    def forward(self, mixture: torch.Tensor, return_aux: bool = False) -> torch.Tensor:
        mix = self._prepare_input(mixture)
        audio_list, audio_aux = self.model(mix)
        est = torch.stack(audio_list, dim=1)
        if est.shape[-1] > mix.shape[-1]:
            est = est[..., : mix.shape[-1]]
        elif est.shape[-1] < mix.shape[-1]:
            est = torch.nn.functional.pad(est, (0, mix.shape[-1] - est.shape[-1]))
        if return_aux or self._return_aux:
            return est, audio_aux
        return est


@register("sepreformer")
def build_sepreformer(sample_rate: int = 16000, num_speakers: int = 2,
                      variant: str = "SepReformer_Base_WSJ0", **kwargs) -> SepReformerAdapter:
    return SepReformerAdapter(sample_rate=sample_rate, num_speakers=num_speakers,
                              variant=variant, **kwargs)
