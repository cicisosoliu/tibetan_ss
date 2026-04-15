"""Scan NICT-Tib1 (Lhasa dialect) and produce per-split speaker manifests.

The NICT-Tib1 corpus, as distributed, is a directory of per-speaker folders
named like ``M001_...`` / ``F002_...``. Exact naming may differ; pass
``--speaker-regex`` to override the default pattern if needed.

Outputs (written under ``<output_root>/manifests/``):

* ``speakers_train.json``  – list of {id, gender, files:[...]}
* ``speakers_val.json``
* ``speakers_test.json``
* ``all_speakers.json``   – union with their split label

The DEMAND noise corpus is scanned in parallel and split by file across
train/val/test with fixed proportions.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None


_DEFAULT_SPEAKER_REGEX = r"^(?P<gender>[MF])(?P<sid>\d+)"  # e.g. M003, F012


def _scan_speakers(root: Path, regex: re.Pattern) -> list[dict]:
    speakers: dict[str, dict] = {}
    for wav in sorted(root.rglob("*.wav")):
        rel_parts = wav.relative_to(root).parts
        # The first component that matches the regex is taken as the speaker id.
        sid = None
        gender = None
        for part in rel_parts:
            m = regex.search(part)
            if m:
                sid = m.group(0)
                gender = m.groupdict().get("gender") or "U"
                break
        if sid is None:
            # fallback: take the first path component
            sid = rel_parts[0]
            gender = "U"
        if sid not in speakers:
            speakers[sid] = {"id": sid, "gender": gender.upper(), "files": []}
        speakers[sid]["files"].append(str(wav))
    return sorted(speakers.values(), key=lambda s: s["id"])


def _split_speakers(speakers: list[dict], cfg: dict, seed: int) -> dict[str, list[dict]]:
    by_gender: dict[str, list[dict]] = defaultdict(list)
    for s in speakers:
        by_gender[s["gender"]].append(s)
    rng = random.Random(seed)
    for g in by_gender:
        rng.shuffle(by_gender[g])

    def take(g: str, n: int) -> list[dict]:
        if n > len(by_gender[g]):
            raise ValueError(
                f"Requested {n} speakers of gender={g}, but only {len(by_gender[g])} are available."
            )
        out = by_gender[g][:n]
        by_gender[g] = by_gender[g][n:]
        return out

    split_counts = cfg
    out: dict[str, list[dict]] = {}
    # train > val > test so training set is assembled first
    for split in ("train", "val", "test"):
        counts = split_counts[split]
        men = take("M", int(counts["male"]))
        women = take("F", int(counts["female"]))
        out[split] = men + women
    return out


def _scan_demand(root: Path) -> list[str]:
    return [str(p) for p in sorted(root.rglob("*.wav"))]


def _split_noise(files: list[str], seed: int, train: float = 0.8, val: float = 0.1) -> dict[str, list[str]]:
    rng = random.Random(seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare NICT-Tib1 + DEMAND manifests.")
    p.add_argument("--config", required=True, help="Path to data config (YAML).")
    p.add_argument("--speaker-regex", default=_DEFAULT_SPEAKER_REGEX)
    p.add_argument("--override-tibetan-root", default=None)
    p.add_argument("--override-noise-root", default=None)
    p.add_argument("--override-output-root", default=None)
    args = p.parse_args()

    if OmegaConf is None:
        raise RuntimeError("omegaconf is required – add it to your environment.")
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    tibetan_root = Path(args.override_tibetan_root or cfg["paths"]["tibetan_root"])
    noise_root = Path(args.override_noise_root or cfg["paths"]["noise_root"])
    output_root = Path(args.override_output_root or cfg["paths"]["output_root"])
    manifests = output_root / cfg["offline"]["manifest_subdir"]
    manifests.mkdir(parents=True, exist_ok=True)

    regex = re.compile(args.speaker_regex)

    speakers = _scan_speakers(tibetan_root, regex)
    print(f"[prepare] found {len(speakers)} speakers in {tibetan_root}")
    for s in speakers:
        print(f"   - {s['id']} ({s['gender']}): {len(s['files'])} files")

    splits = _split_speakers(speakers, cfg["speaker_split"], int(cfg["speaker_split"]["seed"]))
    for split, ss in splits.items():
        out_path = manifests / f"speakers_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(ss, f, indent=2, ensure_ascii=False)
        print(f"[prepare] wrote {out_path} with {len(ss)} speakers")

    all_out = manifests / "all_speakers.json"
    with open(all_out, "w", encoding="utf-8") as f:
        tagged = []
        for split, ss in splits.items():
            for s in ss:
                tagged.append(dict(s, split=split))
        json.dump(tagged, f, indent=2, ensure_ascii=False)
    print(f"[prepare] wrote {all_out}")

    # noise
    noise_files = _scan_demand(noise_root)
    print(f"[prepare] found {len(noise_files)} noise files in {noise_root}")
    noise_splits = _split_noise(noise_files, int(cfg["speaker_split"]["seed"]))
    for split, nf in noise_splits.items():
        out_path = manifests / f"noise_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nf, f, indent=2, ensure_ascii=False)
        print(f"[prepare] wrote {out_path} with {len(nf)} files")


if __name__ == "__main__":
    main()
