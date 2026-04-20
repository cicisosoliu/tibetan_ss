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

    When ``preload=True``, **all** source audio files (speakers + noise) are
    loaded into RAM once at construction time. This eliminates per-step disk
    I/O and can speed up Dynamic Mixing by 10-30x on I/O-bound setups.
    Memory cost: ~7.7 GB for 33.5 h @ 16 kHz + ~5 GB for DEMAND noise.
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
        fixed_length_samples: int | None = None,
        preload: bool = False,
    ) -> None:
        self.split = split
        self.dynamic = dynamic
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.fixed_length_samples = int(fixed_length_samples) if fixed_length_samples else None

        # Audio cache: filepath → float32 numpy array (at target sample rate)
        self._audio_cache: dict[str, np.ndarray] = {}

        if dynamic:
            if speakers is None or mixing_cfg is None:
                raise ValueError("Dynamic mode requires `speakers` and `mixing_cfg`")
            self.speakers = speakers
            self.noise_files = noise_files or []
            self.mixing_cfg = mixing_cfg
            self.simulator = MixtureSimulator(mixing_cfg)
            self._length = int(samples_per_epoch or 10000)
            self._items: list[dict] | None = None
            if preload:
                self._preload_dynamic(mixing_cfg.sample_rate)
        else:
            if manifest_path is None:
                raise ValueError("Offline mode requires `manifest_path`")
            self.manifest_path = Path(manifest_path)
            self._items = _load_manifest(self.manifest_path)
            self._length = len(self._items)
            if preload:
                self._preload_offline()

    # ------------------------------------------------------------------
    # Preloading
    # ------------------------------------------------------------------

    def _preload_dynamic(self, target_sr: int) -> None:
        """Load every speaker + noise wav into ``_audio_cache``."""
        all_files: set[str] = set()
        for spk in self.speakers:
            all_files.update(spk["files"])
        all_files.update(self.noise_files)

        print(f"[preload:{self.split}] loading {len(all_files)} audio files "
              f"into memory @ {target_sr} Hz …", flush=True)
        try:
            from tqdm import tqdm
            iterator = tqdm(sorted(all_files), desc=f"preload {self.split}",
                            unit="file", leave=True)
        except ImportError:
            iterator = sorted(all_files)

        for fpath in iterator:
            wav, _ = read_audio(fpath, target_sr=target_sr)
            self._audio_cache[fpath] = wav

        total_gb = sum(a.nbytes for a in self._audio_cache.values()) / 1e9
        print(f"[preload:{self.split}] done — {len(self._audio_cache)} files, "
              f"{total_gb:.2f} GB", flush=True)

    def _preload_offline(self) -> None:
        """Load every mixture / source wav referenced in the manifest."""
        all_files: set[str] = set()
        for item in self._items:                                # type: ignore[union-attr]
            all_files.add(item["mixture_path"])
            all_files.update(item["source_paths"])
            if item.get("noise_path"):
                all_files.add(item["noise_path"])

        print(f"[preload:{self.split}] loading {len(all_files)} manifest files …",
              flush=True)
        try:
            from tqdm import tqdm
            iterator = tqdm(sorted(all_files), desc=f"preload {self.split}",
                            unit="file", leave=True)
        except ImportError:
            iterator = sorted(all_files)

        for fpath in iterator:
            wav, _ = read_audio(fpath)
            self._audio_cache[fpath] = wav

        total_gb = sum(a.nbytes for a in self._audio_cache.values()) / 1e9
        print(f"[preload:{self.split}] done — {len(self._audio_cache)} files, "
              f"{total_gb:.2f} GB", flush=True)

    # ------------------------------------------------------------------
    def _get_audio(self, path: str, target_sr: int | None = None) -> np.ndarray:
        """Return audio from cache (zero-copy slice) or read from disk."""
        cached = self._audio_cache.get(path)
        if cached is not None:
            return cached                        # MixtureSimulator always
                                                 # creates new arrays downstream
        wav, _ = read_audio(path, target_sr=target_sr)
        return wav

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._length

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._sample_online(idx) if self.dynamic else self._load_offline(idx)
        if self.fixed_length_samples is not None:
            item = _crop_or_pad(item, self.fixed_length_samples, idx, self.dynamic)
        return item

    # ------------------------------------------------------------------
    def _load_offline(self, idx: int) -> dict[str, Any]:
        item = self._items[idx]                               # type: ignore[index]
        sr = int(item["sample_rate"])
        mix = self._get_audio(item["mixture_path"], target_sr=sr)
        s1 = self._get_audio(item["source_paths"][0], target_sr=sr)
        s2 = self._get_audio(item["source_paths"][1], target_sr=sr)
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
        wav_a = self._get_audio(file_a, target_sr=self.mixing_cfg.sample_rate)
        wav_b = self._get_audio(file_b, target_sr=self.mixing_cfg.sample_rate)
        noise_wav = None
        if self.mixing_cfg.noise_enabled and self.noise_files:
            nf = rng.choice(self.noise_files)
            noise_wav = self._get_audio(nf, target_sr=self.mixing_cfg.sample_rate)
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
# Helpers
# ---------------------------------------------------------------------------

def _crop_or_pad(item: dict[str, Any], target_len: int, idx: int, dynamic: bool) -> dict[str, Any]:
    """Crop or right-pad the (mixture, sources) tensors to ``target_len`` samples."""
    mix = item["mixture"]
    src = item["sources"]
    T = mix.shape[-1]
    if T == target_len:
        return item
    if T > target_len:
        offset = 0 if dynamic else (idx * 7919 + 11) % (T - target_len + 1)
        mix = mix[..., offset:offset + target_len]
        src = src[..., offset:offset + target_len]
    else:
        pad = target_len - T
        mix = torch.nn.functional.pad(mix, (0, pad))
        src = torch.nn.functional.pad(src, (0, pad))
    item["mixture"] = mix
    item["sources"] = src
    return item


def _load_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
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
