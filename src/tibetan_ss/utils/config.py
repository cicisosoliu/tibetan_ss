"""Shared Hydra-lite config resolver.

Supports three forms inside ``defaults:``:

* ``_self_``            — merge the current file's remaining fields here
* ``{group: name}``     — load ``configs/<group>/<name>.yaml`` and mount under
                          the ``<group>`` key (leading ``/`` is optional)
* ``"name"``            — inherit a sibling file in the same directory
                          (``<same_dir>/<name>.yaml``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def resolve_defaults(cfg, cfg_path: Path):
    """Recursively expand a ``defaults:`` list, returning a merged OmegaConf."""
    cfg_path = Path(cfg_path)
    root = cfg_path.parent.parent
    if "defaults" not in cfg:
        return cfg
    defaults = cfg.pop("defaults")
    defaults_list = OmegaConf.to_container(defaults, resolve=False)
    merged = OmegaConf.create({})
    for d in defaults_list:
        if isinstance(d, str) and d == "_self_":
            merged = OmegaConf.merge(merged, cfg)
            continue
        if isinstance(d, str):
            sub_path = cfg_path.parent / f"{d}.yaml"
            sub = OmegaConf.load(sub_path)
            sub = resolve_defaults(sub, sub_path)
            merged = OmegaConf.merge(merged, sub)
            continue
        if isinstance(d, dict):
            key, value = next(iter(d.items()))
            key = key.lstrip("/")
            sub_path = root / key / f"{value}.yaml"
            sub = OmegaConf.load(sub_path)
            sub = resolve_defaults(sub, sub_path)
            merged = OmegaConf.merge(merged, OmegaConf.create({key: sub}))
            continue
        raise ValueError(f"Unsupported defaults entry: {d!r}")
    if cfg and "_self_" not in [d for d in defaults_list if isinstance(d, str)]:
        merged = OmegaConf.merge(merged, cfg)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolve its ``defaults:``, and return a plain dict."""
    p = Path(path).resolve()
    cfg = OmegaConf.load(p)
    cfg = resolve_defaults(cfg, p)
    return OmegaConf.to_container(cfg, resolve=True)
