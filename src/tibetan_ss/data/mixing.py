"""Core mixture-simulation primitives.

This module is deliberately framework-agnostic: it takes numpy arrays, sampling
configuration, and an RNG, and returns a mixture + labels dict. Both the
offline mixture-generation script (``scripts/generate_mixtures.py``) and the
online Dynamic-Mixing dataset call into the same functions here.

Generation rules follow the project's PPT (slide 12) and the synthesis
recipe in ``training_data_generation.md``:

* Two speakers (different-speaker pair) from the clean Tibetan corpus.
* Random per-speaker crop of ``segment`` seconds.
* Random ``overlap`` ratio (0..1) drawn from a split-specific distribution.
* Per-speaker RMS normalisation to a common reference level.
* Relative level K ~ U(-5, +5) dB applied to one of the speakers.
* Optional mixing with a DEMAND noise clip at an overall SNR sampled from
  the split-specific range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

_EPS = 1e-8


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2) + _EPS))


def rms_normalize(x: np.ndarray, target_dbfs: float = -25.0) -> np.ndarray:
    target_rms = 10 ** (target_dbfs / 20.0)
    return x * (target_rms / (rms(x) + _EPS))


def apply_level_offset_db(x: np.ndarray, db: float) -> np.ndarray:
    return x * (10.0 ** (db / 20.0))


def _pad_or_crop(x: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[-1] >= length:
        offset = rng.integers(0, x.shape[-1] - length + 1)
        return x[..., offset:offset + length]
    pad = length - x.shape[-1]
    # place original randomly inside the padded buffer
    left = rng.integers(0, pad + 1)
    right = pad - left
    return np.pad(x, ((left, right),) if x.ndim == 1 else ((0, 0), (left, right)))


# ---------------------------------------------------------------------------
# Distribution samplers for overlap / K / SNR
# ---------------------------------------------------------------------------

def _sample_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def sample_overlap(cfg: dict, rng: np.random.Generator) -> float:
    mode = cfg.get("mode", "uniform")
    if mode == "uniform":
        return _sample_uniform(rng, cfg["low"], cfg["high"])
    if mode == "mixture":
        weights = np.asarray([c["weight"] for c in cfg["components"]], dtype=np.float64)
        weights = weights / weights.sum()
        idx = int(rng.choice(len(weights), p=weights))
        return sample_overlap(cfg["components"][idx], rng)
    raise ValueError(f"Unknown overlap mode: {mode}")


def sample_level_diff(cfg: dict, rng: np.random.Generator) -> float:
    return _sample_uniform(rng, cfg["low"], cfg["high"])


def sample_snr(cfg: dict, rng: np.random.Generator) -> float:
    return _sample_uniform(rng, cfg["low"], cfg["high"])


# ---------------------------------------------------------------------------
# Mixture simulation
# ---------------------------------------------------------------------------

@dataclass
class MixingConfig:
    sample_rate: int = 16000
    segment_seconds: float = 3.0
    random_length: bool = True
    min_seconds: float = 2.0
    max_seconds: float = 4.0

    overlap: dict = field(default_factory=lambda: {"mode": "uniform", "low": 0.0, "high": 1.0})
    level_diff_db: dict = field(default_factory=lambda: {"low": -5.0, "high": 5.0})
    snr_db: dict = field(default_factory=lambda: {"low": 2.5, "high": 30.0})

    gender_pairing: Literal["random", "same", "cross"] = "random"
    rms_target_dbfs: float = -25.0

    noise_enabled: bool = True
    noise_prob: float = 1.0

    full_length: bool = False   # if True, keep the natural length (used for the test split)


@dataclass
class MixtureResult:
    """Output of a single mixing event."""

    mixture: np.ndarray                       # shape (T,)
    sources: np.ndarray                       # shape (2, T)
    noise: np.ndarray | None                  # shape (T,) or None
    meta: dict                                # human-readable labels


class MixtureSimulator:
    """Combine two clean Tibetan utterances (+ optional noise) into a 2-speaker mixture.

    Parameters
    ----------
    cfg : MixingConfig
        Sampling configuration.
    """

    def __init__(self, cfg: MixingConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    def segment_length(self, rng: np.random.Generator) -> int:
        if self.cfg.full_length:
            # Signals will be returned at their natural lengths; the caller
            # picks the shorter of the two as the effective length.
            return -1
        if self.cfg.random_length:
            secs = _sample_uniform(rng, self.cfg.min_seconds, self.cfg.max_seconds)
        else:
            secs = self.cfg.segment_seconds
        return int(round(secs * self.cfg.sample_rate))

    # ------------------------------------------------------------------
    def simulate(
        self,
        source_a: np.ndarray,
        source_b: np.ndarray,
        noise: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        *,
        gender_a: str | None = None,
        gender_b: str | None = None,
    ) -> MixtureResult:
        rng = rng if rng is not None else np.random.default_rng()
        L = self.segment_length(rng)

        # --- length handling ------------------------------------------------
        if L <= 0:
            # natural length -> use the shorter of the two (min mode, aligned at 0)
            T = int(min(source_a.shape[-1], source_b.shape[-1]))
            sa = source_a[..., :T].astype(np.float32, copy=False)
            sb = source_b[..., :T].astype(np.float32, copy=False)
        else:
            T = L
            sa = _pad_or_crop(source_a.astype(np.float32, copy=False), T, rng)
            sb = _pad_or_crop(source_b.astype(np.float32, copy=False), T, rng)

        # --- RMS normalise both speakers to a common reference --------------
        sa = rms_normalize(sa, self.cfg.rms_target_dbfs)
        sb = rms_normalize(sb, self.cfg.rms_target_dbfs)

        # --- apply level offset K (dB) to one speaker (random sign) ---------
        K = sample_level_diff(self.cfg.level_diff_db, rng)
        flip = rng.integers(0, 2) == 0
        if flip:
            sa = apply_level_offset_db(sa, +K / 2)
            sb = apply_level_offset_db(sb, -K / 2)
            level_db_a, level_db_b = +K / 2, -K / 2
        else:
            sa = apply_level_offset_db(sa, -K / 2)
            sb = apply_level_offset_db(sb, +K / 2)
            level_db_a, level_db_b = -K / 2, +K / 2

        # --- overlap placement ---------------------------------------------
        overlap = sample_overlap(self.cfg.overlap, rng)
        overlap = float(np.clip(overlap, 0.0, 1.0))
        overlap_len = int(round(T * overlap))
        non_overlap = T - overlap_len                      # total non-overlapping segment
        #    Use layout [A-only | overlap(A+B) | B-only] with random left/right order
        if rng.integers(0, 2) == 0:
            a_start, a_end = 0, overlap_len + non_overlap
            b_start, b_end = non_overlap, T
        else:
            b_start, b_end = 0, overlap_len + non_overlap
            a_start, a_end = non_overlap, T

        src1 = np.zeros(T, dtype=np.float32)
        src2 = np.zeros(T, dtype=np.float32)
        src1[a_start:a_end] = sa[: a_end - a_start]
        src2[b_start:b_end] = sb[: b_end - b_start]

        mix = src1 + src2

        # --- optional noise ------------------------------------------------
        used_noise = None
        snr = None
        if self.cfg.noise_enabled and noise is not None and rng.uniform() < self.cfg.noise_prob:
            n = _pad_or_crop(noise.astype(np.float32, copy=False), T, rng)
            snr = sample_snr(self.cfg.snr_db, rng)
            # scale noise so SNR(mix, n) == snr
            mix_rms = rms(mix)
            n_rms = rms(n)
            if n_rms > 0:
                target_n_rms = mix_rms / (10 ** (snr / 20.0))
                n = n * (target_n_rms / n_rms)
            mix = mix + n
            used_noise = n

        # --- numeric safety: normalise if clipping ---------------------
        peak = float(np.max(np.abs(mix)) + _EPS)
        if peak > 0.99:
            scale = 0.99 / peak
            mix = mix * scale
            src1 = src1 * scale
            src2 = src2 * scale
            if used_noise is not None:
                used_noise = used_noise * scale

        meta = dict(
            overlap_ratio=float(overlap),
            level_diff_db=float(K),
            level_db_a=float(level_db_a),
            level_db_b=float(level_db_b),
            snr_db=None if snr is None else float(snr),
            segment_samples=int(T),
            segment_seconds=float(T / self.cfg.sample_rate),
            gender_a=gender_a,
            gender_b=gender_b,
            gender_pair=_gender_pair_label(gender_a, gender_b),
        )
        return MixtureResult(
            mixture=mix.astype(np.float32),
            sources=np.stack([src1, src2], axis=0).astype(np.float32),
            noise=None if used_noise is None else used_noise.astype(np.float32),
            meta=meta,
        )


# ---------------------------------------------------------------------------
# Gender pairing utilities
# ---------------------------------------------------------------------------

def _gender_pair_label(a: str | None, b: str | None) -> str | None:
    if a is None or b is None:
        return None
    a = a.upper()[0]
    b = b.upper()[0]
    # Order-insensitive so MF == FM
    return "".join(sorted([a, b]))


def pick_speaker_pair(
    speakers: Sequence[dict],
    rng: np.random.Generator,
    pairing: Literal["random", "same", "cross"] = "random",
) -> tuple[dict, dict]:
    """Pick two *different* speakers following the gender-pairing rule."""
    if len(speakers) < 2:
        raise ValueError("Need at least two speakers")
    for _ in range(100):  # bounded loop
        a, b = rng.choice(len(speakers), size=2, replace=False)
        sa, sb = speakers[int(a)], speakers[int(b)]
        if pairing == "random":
            return sa, sb
        if pairing == "same" and sa["gender"] == sb["gender"]:
            return sa, sb
        if pairing == "cross" and sa["gender"] != sb["gender"]:
            return sa, sb
    raise RuntimeError(f"Could not satisfy pairing={pairing} after 100 trials")
