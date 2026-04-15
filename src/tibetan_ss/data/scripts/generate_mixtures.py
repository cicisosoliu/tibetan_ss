"""Offline mixture generation.

Reads the speaker/noise manifests produced by ``prepare_nict_tib1.py`` and
writes:

* ``<output_root>/mixtures/<split>/<mix_id>/mixture.wav``
* ``<output_root>/mixtures/<split>/<mix_id>/s1.wav``
* ``<output_root>/mixtures/<split>/<mix_id>/s2.wav``
* ``<output_root>/mixtures/<split>/<mix_id>/noise.wav`` (if any)
* ``<output_root>/manifests/<split>.json``  (dataset manifest)

Mixture IDs follow the convention suggested in slide 11 of the PPT:
``mix_{split}_{gender_pair}_ov{overlap*100:03d}_n{noise_flag}_ns{snr*10:03d}_rlm{K*10}_{idx}``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None

from tibetan_ss.data.mixing import MixingConfig, MixtureSimulator, pick_speaker_pair
from tibetan_ss.utils.io import read_audio, write_audio


def _mix_id(split: str, idx: int, meta: dict) -> str:
    gp = meta.get("gender_pair") or "UU"
    ov = int(round(float(meta["overlap_ratio"]) * 100))
    snr_tag = "n0" if meta.get("snr_db") is None else f"n1"
    ns_tag = "ns0" if meta.get("snr_db") is None else f"ns{int(round(meta['snr_db'] * 10)):03d}"
    K = float(meta["level_diff_db"])
    sign = "p" if K >= 0 else "m"
    rlm = f"rl{sign}{int(round(abs(K) * 10)):02d}"
    return f"mix_{split}_{gp}_ov{ov:03d}_{snr_tag}_{ns_tag}_{rlm}_{idx:06d}"


def _build_mixing_cfg(cfg: dict, split: str, sr: int) -> MixingConfig:
    seg = cfg["segment"]
    mix = cfg["mixing"]
    seg_seconds = float(seg[split]) if seg[split] is not None else 0.0
    return MixingConfig(
        sample_rate=sr,
        segment_seconds=seg_seconds,
        random_length=bool(seg.get("random_length", True)) and seg[split] is not None,
        min_seconds=float(seg["min_seconds"]),
        max_seconds=float(seg["max_seconds"]),
        overlap=mix["overlap"][split],
        level_diff_db=mix["level_diff_db"],
        snr_db=mix["snr_db"][split],
        gender_pairing=str(mix.get("gender_pairing", "random")),
        rms_target_dbfs=float(mix.get("rms_target_dbfs", -25.0)),
        noise_enabled=bool(cfg["noise"]["enabled"]),
        noise_prob=float(cfg["noise"].get("prob_apply", 1.0)),
        full_length=(seg[split] is None),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Generate offline 2-speaker mixtures.")
    p.add_argument("--config", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--override-output-root", default=None)
    args = p.parse_args()

    if OmegaConf is None:
        raise RuntimeError("omegaconf is required")
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    sr = int(cfg["sample_rate"])
    out_root = Path(args.override_output_root or cfg["paths"]["output_root"])
    mix_root = out_root / cfg["offline"]["output_subdir"]
    manifests = out_root / cfg["offline"]["manifest_subdir"]
    manifests.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        speakers_path = manifests / f"speakers_{split}.json"
        noise_path = manifests / f"noise_{split}.json"
        with open(speakers_path, "r", encoding="utf-8") as f:
            speakers = json.load(f)
        with open(noise_path, "r", encoding="utf-8") as f:
            noise_files = json.load(f)

        mixing_cfg = _build_mixing_cfg(cfg, split, sr)
        simulator = MixtureSimulator(mixing_cfg)
        n = int(cfg["offline"]["num_mixtures"][split])
        seed = int(cfg["offline"]["seed"]) + _SPLIT_SEED[split]
        rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        items = []
        split_dir = mix_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for idx in tqdm(range(n), desc=f"generate {split}"):
            sa, sb = pick_speaker_pair(speakers, rng, mixing_cfg.gender_pairing)
            file_a = py_rng.choice(sa["files"])
            file_b = py_rng.choice(sb["files"])
            wav_a, _ = read_audio(file_a, target_sr=sr)
            wav_b, _ = read_audio(file_b, target_sr=sr)
            noise_wav = None
            if mixing_cfg.noise_enabled and noise_files:
                nf = py_rng.choice(noise_files)
                noise_wav, _ = read_audio(nf, target_sr=sr)
            result = simulator.simulate(
                wav_a, wav_b, noise_wav, rng=rng,
                gender_a=sa.get("gender"), gender_b=sb.get("gender"),
            )
            mid = _mix_id(split, idx, result.meta)
            mdir = split_dir / mid
            mdir.mkdir(parents=True, exist_ok=True)
            write_audio(mdir / "mixture.wav", result.mixture, sr, subtype=cfg["offline"]["audio_subtype"])
            write_audio(mdir / "s1.wav",      result.sources[0], sr, subtype=cfg["offline"]["audio_subtype"])
            write_audio(mdir / "s2.wav",      result.sources[1], sr, subtype=cfg["offline"]["audio_subtype"])
            if result.noise is not None:
                write_audio(mdir / "noise.wav", result.noise, sr, subtype=cfg["offline"]["audio_subtype"])
            items.append({
                "id":            mid,
                "mixture_path":  str(mdir / "mixture.wav"),
                "source_paths":  [str(mdir / "s1.wav"), str(mdir / "s2.wav")],
                "noise_path":    str(mdir / "noise.wav") if result.noise is not None else None,
                "speaker_ids":   [sa["id"], sb["id"]],
                "speaker_files": [file_a, file_b],
                "sample_rate":   sr,
                "meta":          result.meta,
            })

        manifest_path = manifests / f"{split}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"split": split, "items": items, "sample_rate": sr}, f, indent=2, ensure_ascii=False)
        print(f"[mix] wrote {manifest_path} with {len(items)} items")


_SPLIT_SEED = {"train": 101, "val": 202, "test": 303}


if __name__ == "__main__":
    main()
