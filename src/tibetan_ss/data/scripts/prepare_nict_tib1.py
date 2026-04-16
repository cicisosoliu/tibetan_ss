"""Scan NICT-Tib1 + DEMAND and produce per-split speaker / noise manifests.

Two speaker-id discovery modes are supported:

* **nict_tib1** (recommended for this corpus) – speaker directories are pure
  numeric IDs (``006/``, ``007/`` …) with no gender hint in the name, so an
  external ``gender_map`` YAML is required. Layout expected:

      <tibetan_root>/data/<speaker_id>/<session>/<utt>.wav

* **regex** (legacy / WSJ-style) – a regex like ``^(?P<gender>[MF])(?P<sid>\\d+)``
  is matched against path components until a speaker id + gender is found.

The noise corpus (DEMAND) is scanned in parallel and split into train / val /
test by file at fixed 80/10/10 proportions. If ``noise.enabled`` is False or
``noise_root`` is empty, the noise step is skipped.
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

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


_DEFAULT_SPEAKER_REGEX = r"^(?P<gender>[MF])(?P<sid>\d+)"


# ---------------------------------------------------------------------------
# Speaker discovery
# ---------------------------------------------------------------------------

def _scan_speakers_regex(root: Path, regex: re.Pattern) -> list[dict]:
    speakers: dict[str, dict] = {}
    for wav in sorted(root.rglob("*.wav")):
        rel_parts = wav.relative_to(root).parts
        sid = None
        gender = None
        for part in rel_parts:
            m = regex.search(part)
            if m:
                sid = m.group(0)
                gender = m.groupdict().get("gender") or "U"
                break
        if sid is None:
            sid = rel_parts[0]
            gender = "U"
        if sid not in speakers:
            speakers[sid] = {"id": sid, "gender": gender.upper(), "files": []}
        speakers[sid]["files"].append(str(wav))
    return sorted(speakers.values(), key=lambda s: s["id"])


def _scan_speakers_nict_tib1(root: Path, data_subdir: str, gender_map: dict[str, str]) -> list[dict]:
    """Scan NICT-Tib1 layout: ``<root>/<data_subdir>/<spk>/<session>/<utt>.wav``.

    Speaker ID = the first directory under ``data_subdir``. Gender is looked
    up from ``gender_map`` (required).
    """
    speakers_root = root / data_subdir if data_subdir else root
    if not speakers_root.exists():
        raise FileNotFoundError(
            f"NICT-Tib1 data dir not found: {speakers_root}. Did you set "
            f"`paths.tibetan_root` correctly, and does it contain a `{data_subdir}/` subfolder?"
        )
    speakers: dict[str, dict] = {}
    for wav in sorted(speakers_root.rglob("*.wav")):
        rel_parts = wav.relative_to(speakers_root).parts
        if not rel_parts:
            continue
        sid = rel_parts[0]                              # e.g. "006"
        if sid not in speakers:
            gender = gender_map.get(sid, "U").upper()
            if gender not in ("M", "F"):
                # Unknown speaker – skip so it doesn't pollute the split
                continue
            speakers[sid] = {"id": sid, "gender": gender, "files": []}
        speakers[sid]["files"].append(str(wav))
    missing = set(gender_map) - set(speakers)
    if missing:
        print(f"[prepare] WARN gender_map has {len(missing)} speakers not found on disk: {sorted(missing)}")
    return sorted(speakers.values(), key=lambda s: s["id"])


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _split_speakers(speakers: list[dict], cfg: dict, seed: int) -> dict[str, list[dict]]:
    by_gender: dict[str, list[dict]] = defaultdict(list)
    for s in speakers:
        by_gender[s["gender"]].append(s)
    rng = random.Random(seed)
    for g in by_gender:
        rng.shuffle(by_gender[g])

    def take(g: str, n: int) -> list[dict]:
        pool = by_gender.get(g, [])
        if n > len(pool):
            raise ValueError(
                f"Requested {n} speakers of gender={g}, but only {len(pool)} are available. "
                f"Pool: {[s['id'] for s in pool]}"
            )
        out = pool[:n]
        by_gender[g] = pool[n:]
        return out

    out: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        counts = cfg[split]
        men = take("M", int(counts["male"]))
        women = take("F", int(counts["female"]))
        out[split] = men + women
    return out


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gender_map(path: Path) -> dict[str, str]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load a gender map.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {str(k): str(v).upper() for k, v in data.items()}


def _resolve_gender_map_path(cfg: dict, config_path: Path, cli_override: str | None) -> Path | None:
    if cli_override:
        p = Path(cli_override)
    else:
        sp = cfg.get("speaker_source") or {}
        gm = sp.get("gender_map")
        if not gm:
            return None
        # Strip hydra placeholder and resolve relative to the config file's parent.
        gm = str(gm).replace("${hydra:runtime.cwd}", "")
        p = Path(gm)
        if not p.is_absolute():
            # Try two candidates: cwd-relative and config-relative.
            for candidate in (Path.cwd() / p, config_path.parent / p):
                if candidate.exists():
                    p = candidate
                    break
    return p if p.exists() else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Prepare NICT-Tib1 + DEMAND manifests.")
    p.add_argument("--config", required=True, help="Path to data config (YAML).")
    p.add_argument("--speaker-regex", default=_DEFAULT_SPEAKER_REGEX,
                   help="Regex used when speaker_source.kind != 'nict_tib1'.")
    p.add_argument("--gender-map", default=None,
                   help="Override gender map path (YAML: {speaker_id: M|F|U}).")
    p.add_argument("--override-tibetan-root", default=None)
    p.add_argument("--override-noise-root", default=None)
    p.add_argument("--override-output-root", default=None)
    args = p.parse_args()

    if OmegaConf is None:
        raise RuntimeError("omegaconf is required – add it to your environment.")
    config_path = Path(args.config).resolve()
    from tibetan_ss.utils.config import load_config
    cfg = load_config(config_path)

    tibetan_root = Path(args.override_tibetan_root or cfg["paths"]["tibetan_root"])
    noise_root_raw = args.override_noise_root or cfg["paths"].get("noise_root") or ""
    noise_root = Path(noise_root_raw) if noise_root_raw else None
    output_root = Path(args.override_output_root or cfg["paths"]["output_root"])
    manifests = output_root / cfg["offline"]["manifest_subdir"]
    manifests.mkdir(parents=True, exist_ok=True)

    # ---- Speakers ------------------------------------------------------
    source_kind = (cfg.get("speaker_source") or {}).get("kind", "regex")
    if source_kind == "nict_tib1":
        gmap_path = _resolve_gender_map_path(cfg, config_path, args.gender_map)
        if gmap_path is None:
            raise FileNotFoundError(
                "speaker_source.kind=nict_tib1 but no gender_map could be located. "
                "Pass --gender-map explicitly or set `speaker_source.gender_map` in the config."
            )
        print(f"[prepare] loading gender map from {gmap_path}")
        gender_map = _load_gender_map(gmap_path)
        data_subdir = (cfg.get("speaker_source") or {}).get("data_subdir", "data")
        speakers = _scan_speakers_nict_tib1(tibetan_root, data_subdir, gender_map)
    else:
        regex = re.compile(args.speaker_regex)
        speakers = _scan_speakers_regex(tibetan_root, regex)

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

    # ---- Noise (optional) ---------------------------------------------
    noise_enabled = bool(cfg.get("noise", {}).get("enabled", False))
    if not noise_enabled or noise_root is None or not noise_root.exists():
        reason = ("noise.enabled=false" if not noise_enabled
                  else "DEMAND root not found or unset")
        print(f"[prepare] skipping noise scan ({reason}); writing empty noise manifests.")
        for split in ("train", "val", "test"):
            out_path = manifests / f"noise_{split}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            print(f"[prepare] wrote empty {out_path}")
        return

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
