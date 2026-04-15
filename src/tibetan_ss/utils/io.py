from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml


def read_audio(path: str | Path, target_sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
    """Read an audio file as float32 numpy.

    Returns
    -------
    wav : (T,) float32 when mono else (T, C) float32
    sr  : int sampling rate
    """
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1 and mono:
        wav = wav.mean(axis=1)
    if target_sr is not None and sr != target_sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
        sr = target_sr
    return wav.astype(np.float32, copy=False), sr


def write_audio(path: str | Path, wav: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr, subtype=subtype)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str | Path, obj: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
