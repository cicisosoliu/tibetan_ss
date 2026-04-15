"""Registry so that ``configs/model/*.yaml`` can spawn models by name."""

from __future__ import annotations

import importlib
from typing import Callable, Any

from .base import BaseSeparator

_REGISTRY: dict[str, Callable[..., BaseSeparator]] = {}


def register(name: str):
    def _inner(fn: Callable[..., BaseSeparator]):
        if name in _REGISTRY:
            raise ValueError(f"Model `{name}` already registered")
        _REGISTRY[name] = fn
        return fn
    return _inner


def list_models() -> list[str]:
    return sorted(_REGISTRY.keys())


def build_model(cfg: dict) -> BaseSeparator:
    """``cfg`` must be a mapping with at least ``name: str``.

    Extra keys are forwarded to the constructor.
    """
    if "name" not in cfg:
        raise KeyError("Model config must include `name`")
    name = cfg["name"]
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model `{name}`. Registered: {list_models()}")
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return _REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------------
# Side-effect: import the builtin models so they self-register. Optional ones
# are best-effort – if a dependency is missing we log and move on.
# ---------------------------------------------------------------------------

def _safe_import(mod: str) -> Any | None:
    try:
        return importlib.import_module(mod)
    except Exception as e:  # pragma: no cover
        print(f"[registry] skipped {mod}: {e}")
        return None


_safe_import("tibetan_ss.models.identity")
_safe_import("tibetan_ss.models.proposed.model")
_safe_import("tibetan_ss.models.tiger")
_safe_import("tibetan_ss.models.sepreformer")
_safe_import("tibetan_ss.models.dual_path_mamba")
_safe_import("tibetan_ss.models.mossformer2")
_safe_import("tibetan_ss.models.dip_frontend")
