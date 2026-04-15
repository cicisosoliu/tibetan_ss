"""PyTorch Dataset implementations: offline manifest + dynamic mixing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.io import read_audio
from .mixing import MixingConfig, MixtureSimulator, pick_speaker_pair


class TibetanMixDataset(Dataset):
    """Serves 2-speaker mixtures for training / evaluation.

    Two operating modes are supported:

    * **Offline / static** (``dynamic=False``) – reads a manifest produced by
      ``scripts/data/generate_mixtures.py`` and simply loads the pre-rendered
      mixture/source waveforms. This mode is reproducible bit-for-bit.

    * **Dynamic mixing** (``dynamic=True``) – samples two source speakers on
      every call and re-synthesises a mixture in memory. Useful for training
      runs that want extra data diversity at the cost of reproducibility.
    """

    def __init__(
        self,
        split: str,
        *,
        manifest_path: str | Path | None = None,
        speakers: list[dict] | None = None,
        noise_files: list[str] | None = None,
        mixing_cfg: MixingConfig | None = None,
        dynamic: bool = False,
        samples_per_epoch: int | None = None,
        seed: int = 0,
    ) -> None:
        self.split = split
        self.dynamic = dynamic
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        if dynamic:
            if speakers is None or mixing_cfg is None:
                raise ValueError("Dynamic mode requires `speakers` and `mixing_cfg`")
            self.speakers = speakers
            self.noise_files = noise_files or []
            self.mixing_cfg = mixing_cfg
            self.simulator = MixtureSimulator(mixing_cfg)
            self._length = int(samples_per_epoch or 10000)
            self._items: list[dict] | None = None
        else:
            if manifest_path is None:
                raise ValueError("Offline mode requires `manifest_path`")
            self.manifest_path = Path(manifest_path)
            self._items = _load_manifest(self.manifest_path)
            self._length = len(self._items)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._length

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.dynamic:
            return self._sample_online(idx)
        return self._load_offline(idx)

    # ------------------------------------------------------------------
    def _load_offline(self, idx: int) -> dict[str, Any]:
        item = self._items[idx]                               # type: ignore[index]
        sr = int(item["sample_rate"])
        mix, _ = read_audio(item["mixture_path"], target_sr=sr)
        s1, _ = read_audio(item["source_paths"][0], target_sr=sr)
        s2, _ = read_audio(item["source_paths"][1], target_sr=sr)
        return {
            "mixture":  torch.from_numpy(mix),
            "sources":  torch.from_numpy(np.stack([s1, s2], axis=0)),
            "meta":     item.get("meta", {}),
            "id":       item.get("id", str(idx)),
            "sample_rate": sr,
        }

    # ------------------------------------------------------------------
    def _sample_online(self, idx: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        sa, sb = pick_speaker_pair(self.speakers, rng, self.mixing_cfg.gender_pairing)
        file_a = rng.choice(sa["files"])
        file_b = rng.choice(sb["files"])
        wav_a, _ = read_audio(file_a, target_sr=self.mixing_cfg.sample_rate)
        wav_b, _ = read_audio(file_b, target_sr=self.mixing_cfg.sample_rate)
        noise_wav = None
        if self.mixing_cfg.noise_enabled and self.noise_files:
            nf = rng.choice(self.noise_files)
            noise_wav, _ = read_audio(nf, target_sr=self.mixing_cfg.sample_rate)
        result = self.simulator.simulate(
            wav_a, wav_b, noise_wav, rng=rng,
            gender_a=sa.get("gender"), gender_b=sb.get("gender"),
        )
        return {
            "mixture": torch.from_numpy(result.mixture),
            "sources": torch.from_numpy(result.sources),
            "meta":    result.meta | {"speaker_a": sa["id"], "speaker_b": sb["id"]},
            "id":      f"dm_{self.split}_{idx:08d}",
            "sample_rate": self.mixing_cfg.sample_rate,
        }


# ---------------------------------------------------------------------------
# Manifest IO
# ---------------------------------------------------------------------------

def _load_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either {items: [...]} or a bare list
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognised manifest format: {path}")


# ---------------------------------------------------------------------------
# Collate: right-pads to the longest clip in the batch (for variable-length eval)
# ---------------------------------------------------------------------------

def collate_variable_length(batch: list[dict]) -> dict[str, Any]:
    max_T = max(b["mixture"].shape[-1] for b in batch)
    mixtures = torch.zeros(len(batch), max_T)
    sources = torch.zeros(len(batch), 2, max_T)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    ids = []
    metas = []
    for i, b in enumerate(batch):
        T = b["mixture"].shape[-1]
        mixtures[i, :T] = b["mixture"]
        sources[i, :, :T] = b["sources"]
        lengths[i] = T
        ids.append(b["id"])
        metas.append(b.get("meta", {}))
    return {
        "mixture": mixtures,
        "sources": sources,
        "length":  lengths,
        "id":      ids,
        "meta":    metas,
        "sample_rate": batch[0]["sample_rate"],
    }
