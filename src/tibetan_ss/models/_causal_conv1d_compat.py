"""Compatibility shim for causal-conv1d >= 1.4.

Mamba-TasNet's vendored ``selective_scan_interface.py`` calls
``causal_conv1d_cuda.causal_conv1d_fwd`` with the **old** 5-argument signature
(causal-conv1d <= 1.3). Since version 1.4 the C extension changed to 8 args
for fwd and 9 args for bwd, causing a hard TypeError.

This module provides a drop-in ``causal_conv1d_cuda`` replacement that
translates old calls → new calls. We inject it into ``sys.modules`` before
the vendored code is imported, so it sees our shim instead of the real
extension.

Usage (from the adapter)::

    from tibetan_ss.models._causal_conv1d_compat import install_compat_shim
    install_compat_shim()        # call ONCE before importing Mamba-TasNet modules
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _needs_shim() -> bool:
    """Return True if the installed causal_conv1d_cuda has the new (>=1.4) API."""
    try:
        import causal_conv1d_cuda as _real
        import torch
        import inspect
        # The new API's fwd takes 8 positional args; the old takes 5.
        # We detect by trying a dummy call pattern (will TypeError on new API).
        # Simpler: just inspect the binding's docstring / arg count.
        # pybind11 functions have __doc__ with the signature.
        doc = getattr(_real.causal_conv1d_fwd, "__doc__", "") or ""
        # Count "arg" occurrences in pybind docstring as a proxy for arg count
        if doc.count("arg") >= 7:      # new API has 8 args → ≥7 "argN" mentions
            return True
        # Fallback: if we can't tell from doc, check the module version
        try:
            import causal_conv1d
            version = tuple(int(x) for x in causal_conv1d.__version__.split(".")[:2])
            return version >= (1, 4)
        except Exception:
            pass
        return False
    except ImportError:
        return False


class _CompatModule(ModuleType):
    """A fake ``causal_conv1d_cuda`` that wraps new-API calls for old callers."""

    def __init__(self, real_module):
        super().__init__("causal_conv1d_cuda")
        self._real = real_module

    # ------------------------------------------------------------------
    # fwd: old (x, w, b, seq_idx, silu) → new (x, w, b, seq_idx, init, out, final, silu)
    # ------------------------------------------------------------------
    @staticmethod
    def _make_fwd(real_fwd):
        import torch

        def causal_conv1d_fwd(x, weight, bias, seq_idx, silu_activation):
            output = torch.empty_like(x)
            real_fwd(x, weight, bias, seq_idx, None, output, None, silu_activation)
            return output

        return causal_conv1d_fwd

    # ------------------------------------------------------------------
    # bwd: old (x, w, b, dout, seq_idx, dx, silu) → new adds init_states, dfinal_states
    # ------------------------------------------------------------------
    @staticmethod
    def _make_bwd(real_bwd):
        def causal_conv1d_bwd(x, weight, bias, dout, seq_idx, dx, silu_activation):
            return real_bwd(x, weight, bias, dout, seq_idx, None, None, dx, silu_activation)

        return causal_conv1d_bwd

    def __getattr__(self, name):
        if name == "causal_conv1d_fwd":
            return self._make_fwd(self._real.causal_conv1d_fwd)
        if name == "causal_conv1d_bwd":
            return self._make_bwd(self._real.causal_conv1d_bwd)
        # Everything else (causal_conv1d_update, etc.) passes through
        return getattr(self._real, name)


_installed = False


def install_compat_shim() -> None:
    """Install the shim into ``sys.modules['causal_conv1d_cuda']``.

    Safe to call multiple times (no-op after the first).
    """
    global _installed
    if _installed:
        return
    if not _needs_shim():
        _installed = True
        return
    import causal_conv1d_cuda as _real
    shim = _CompatModule(_real)
    sys.modules["causal_conv1d_cuda"] = shim
    _installed = True
