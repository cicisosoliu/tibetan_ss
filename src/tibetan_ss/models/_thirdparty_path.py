"""Add ``third_party/<repo>`` to ``sys.path`` so upstream model code runs
without modification.

We deliberately keep this simple — ``third_party/`` lives at the repo root,
alongside ``src/``. Adapter modules call :func:`register_thirdparty` near the
top of the file before importing the upstream module.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_THIRDPARTY = _ROOT / "third_party"


def register_thirdparty(repo_name: str) -> Path:
    path = _THIRDPARTY / repo_name
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {path} – did you clone the repo? See third_party/README.md"
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path
