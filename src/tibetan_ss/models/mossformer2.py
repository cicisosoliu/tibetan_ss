"""Adapter for MossFormer2 (Zhao et al., ICASSP 2024).

Official standalone code: https://github.com/alibabasglab/MossFormer2
(MossFormer2 is NOT part of SpeechBrain core — it's maintained by Alibaba DAMO.)

Clone the repo into ``third_party/``:

.. code-block:: bash

    cd third_party
    git clone --depth 1 https://github.com/alibabasglab/MossFormer2.git

Extra pip dependency:

.. code-block:: bash

    pip install rotary_embedding_torch huggingface_hub
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ._thirdparty_path import register_thirdparty
from .base import BaseSeparator
from .registry import register


# Default config — matches ``mossformer2_librimix_2spk`` from upstream but
# lets us override num_spks and sample_rate via our YAML.
_DEFAULT_CONFIG = {
    "model_type": "mossformer2",
    "config_name": "mossformer2-tibetan-2spk",
    "sample_rate": 8000,
    "encoder_kernel_size": 16,
    "encoder_out_nchannels": 512,
    "encoder_in_nchannels": 1,
    "masknet_numspks": 2,
    "masknet_chunksize": 250,
    "masknet_numlayers": 1,
    "masknet_norm": "ln",
    "masknet_useextralinearlayer": False,
    "masknet_extraskipconnection": True,
    "intra_numlayers": 24,
    "intra_nhead": 8,
    "intra_dffn": 1024,
    "intra_dropout": 0,
    "intra_use_positional": True,
    "intra_norm_before": True,
}


class MossFormer2Adapter(BaseSeparator):
    """Wraps ``Mossformer2Wrapper`` from the official standalone repo.

    The upstream forward returns ``(B, T, num_spks)``; we transpose to our
    standard ``(B, num_spks, T)`` contract.
    """

    def __init__(self, sample_rate: int = 16000, num_speakers: int = 2,
                 variant: str = "L", **kwargs: Any):
        super().__init__(num_speakers=num_speakers, sample_rate=sample_rate)
        repo_path = register_thirdparty("MossFormer2")
        import sys
        standalone = str(repo_path / "MossFormer2_standalone")
        if standalone not in sys.path:
            sys.path.insert(0, standalone)

        from model.mossformer2 import Mossformer2Wrapper

        cfg = dict(_DEFAULT_CONFIG)
        cfg["masknet_numspks"] = num_speakers
        cfg["sample_rate"] = sample_rate
        if sample_rate >= 16000:
            cfg["encoder_kernel_size"] = 32
            cfg["masknet_chunksize"] = 400
        for k, v in kwargs.items():
            if k in cfg:
                cfg[k] = v
        self.model = Mossformer2Wrapper(config=cfg)
        self.variant = variant

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        mix = self._prepare_input(mixture)            # (B, T)
        out = self.model(mix)                          # (B, T, num_spks)
        out = out.transpose(1, 2).contiguous()         # (B, num_spks, T)
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
