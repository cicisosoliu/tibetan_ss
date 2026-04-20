"""Compatibility shim for causal-conv1d API changes.

Mamba-TasNet's vendored ``selective_scan_interface.py`` calls
``causal_conv1d_cuda.causal_conv1d_fwd`` with the **old** 5-argument signature
(causal-conv1d <= 1.1). Newer versions changed the C extension:

- 7-arg fwd (causal-conv1d ~1.2-1.3):
    ``(x, w, b, seq_idx, initial_states, final_states, silu) -> Tensor``
- 8-arg fwd (causal-conv1d >= 1.4):
    ``(x, w, b, seq_idx, initial_states, output, final_states, silu) -> void``

This module provides a drop-in ``causal_conv1d_cuda`` replacement that
translates old 5-arg calls → whatever the installed version expects.
We inject it into ``sys.modules`` before the vendored code is imported.

Usage (from the adapter)::

    from tibetan_ss.models._causal_conv1d_compat import install_compat_shim
    install_compat_shim()        # call ONCE before importing Mamba-TasNet modules
"""

from __future__ import annotations

import sys
from types import ModuleType


def _count_pybind_args(func) -> int:
    """Count positional args of a pybind11 function from its docstring."""
    doc = getattr(func, "__doc__", "") or ""
    return doc.count("arg")


def _needs_shim() -> bool:
    """Return True if the installed causal_conv1d_cuda has a different API."""
    try:
        import causal_conv1d_cuda as _real
        n = _count_pybind_args(_real.causal_conv1d_fwd)
        return n > 5  # Mamba-TasNet expects 5 args
    except ImportError:
        return False


class _CompatModule(ModuleType):
    """A fake ``causal_conv1d_cuda`` that wraps new-API calls for old callers."""

    def __init__(self, real_module):
        super().__init__("causal_conv1d_cuda")
        self._real = real_module
        self._fwd_nargs = _count_pybind_args(real_module.causal_conv1d_fwd)
        self._bwd_nargs = _count_pybind_args(getattr(real_module, "causal_conv1d_bwd", None) or (lambda: None))

    # ------------------------------------------------------------------
    # fwd wrappers: old 5-arg → new 7-arg or 8-arg
    # ------------------------------------------------------------------
    @staticmethod
    def _make_fwd_7(real_fwd):
        """Wrap 7-arg: (x, w, b, seq_idx, initial_states, final_states, silu) -> Tensor"""
        def causal_conv1d_fwd(x, weight, bias, seq_idx, silu_activation):
            return real_fwd(x, weight, bias, seq_idx, None, None, silu_activation)
        return causal_conv1d_fwd

    @staticmethod
    def _make_fwd_8(real_fwd):
        """Wrap 8-arg: (x, w, b, seq_idx, initial_states, output, final_states, silu) -> void"""
        import torch

        def causal_conv1d_fwd(x, weight, bias, seq_idx, silu_activation):
            output = torch.empty_like(x)
            real_fwd(x, weight, bias, seq_idx, None, output, None, silu_activation)
            return output
        return causal_conv1d_fwd

    # ------------------------------------------------------------------
    # bwd wrappers: old 7-arg → new 9-arg or 10-arg
    # Old: (x, w, b, dout, seq_idx, dx, silu)
    # 9-arg: (x, w, b, dout, seq_idx, initial_states, dfinal_states, dx, silu)
    # 10-arg: (x, w, b, dout, seq_idx, initial_states, dfinal_states, dx, silu, return_dinitial_states)
    # ------------------------------------------------------------------
    @staticmethod
    def _make_bwd_9(real_bwd):
        def causal_conv1d_bwd(x, weight, bias, dout, seq_idx, dx, silu_activation):
            result = real_bwd(x, weight, bias, dout, seq_idx, None, None, dx, silu_activation)
            # Old API returns (dx, dweight, dbias); new may return extra grads
            return result[0], result[1], result[2]
        return causal_conv1d_bwd

    @staticmethod
    def _make_bwd_10(real_bwd):
        def causal_conv1d_bwd(x, weight, bias, dout, seq_idx, dx, silu_activation):
            result = real_bwd(x, weight, bias, dout, seq_idx, None, None, dx, silu_activation, False)
            return result[0], result[1], result[2]
        return causal_conv1d_bwd

    def __getattr__(self, name):
        if name == "causal_conv1d_fwd":
            if self._fwd_nargs >= 8:
                fn = self._make_fwd_8(self._real.causal_conv1d_fwd)
            else:
                fn = self._make_fwd_7(self._real.causal_conv1d_fwd)
            self.__dict__[name] = fn  # cache
            return fn
        if name == "causal_conv1d_bwd":
            if self._bwd_nargs >= 10:
                fn = self._make_bwd_10(self._real.causal_conv1d_bwd)
            elif self._bwd_nargs > 7:
                fn = self._make_bwd_9(self._real.causal_conv1d_bwd)
            else:
                fn = getattr(self._real, name)
            self.__dict__[name] = fn
            return fn
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
