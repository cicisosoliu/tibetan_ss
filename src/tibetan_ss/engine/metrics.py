"""Evaluation metrics: SI-SDR, SI-SDRi, PESQ, STOI.

All batch-level helpers accept tensors on any device and return CPU floats
(SI-SDR is vectorised; PESQ/STOI fall back to per-utterance CPU numpy).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from ..losses.sisdr import si_sdr


@torch.no_grad()
def si_sdr_batch(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-source SI-SDR, shape (B, C)."""
    return si_sdr(est, ref)


@torch.no_grad()
def si_sdri_batch(est: torch.Tensor, ref: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
    """SI-SDR improvement vs. the mixture (expanded to every reference).

    ``mixture`` has shape (B, T) and is broadcast to (B, C, T).
    """
    B, C, T = est.shape
    base = si_sdr(mixture.unsqueeze(1).expand(-1, C, -1), ref)
    return si_sdr_batch(est, ref) - base


@torch.no_grad()
def pesq_batch(est: torch.Tensor, ref: torch.Tensor, sample_rate: int,
                mode: str = "wb") -> torch.Tensor:
    """Wideband (16k) or narrowband (8k) PESQ. Returns (B, C) tensor of PESQ values.

    A clip with numerical issues (or that is too short) returns NaN instead of
    raising, so downstream code must call ``nanmean`` to aggregate.
    """
    try:
        from pesq import pesq as _pesq
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("install `pesq` to enable PESQ metric") from e

    if mode == "wb" and sample_rate != 16000:
        # PESQ-wb requires 16k
        mode = "nb"
    if mode == "nb" and sample_rate != 8000 and sample_rate != 16000:
        # we only support 8k / 16k PESQ
        return torch.full(est.shape[:2], float("nan"))

    est_np = est.detach().cpu().numpy()
    ref_np = ref.detach().cpu().numpy()
    B, C, _ = est_np.shape
    out = np.full((B, C), np.nan, dtype=np.float32)
    for b in range(B):
        for c in range(C):
            try:
                out[b, c] = _pesq(sample_rate, ref_np[b, c], est_np[b, c], mode)
            except Exception:
                out[b, c] = np.nan
    return torch.from_numpy(out)


@torch.no_grad()
def stoi_batch(est: torch.Tensor, ref: torch.Tensor, sample_rate: int,
               extended: bool = False) -> torch.Tensor:
    try:
        from pystoi.stoi import stoi as _stoi
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("install `pystoi` to enable STOI metric") from e
    est_np = est.detach().cpu().numpy()
    ref_np = ref.detach().cpu().numpy()
    B, C, _ = est_np.shape
    out = np.full((B, C), np.nan, dtype=np.float32)
    for b in range(B):
        for c in range(C):
            try:
                out[b, c] = _stoi(ref_np[b, c], est_np[b, c], sample_rate, extended=extended)
            except Exception:
                out[b, c] = np.nan
    return torch.from_numpy(out)


def _safe_mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    t = x.detach().cpu().float()
    m = torch.isfinite(t)
    if not m.any():
        return float("nan")
    return float(t[m].mean().item())


@torch.no_grad()
def evaluate_batch(
    est: torch.Tensor,
    ref: torch.Tensor,
    mixture: torch.Tensor,
    sample_rate: int,
    metric_list: Iterable[str] = ("si_sdr", "si_sdri", "pesq_wb", "stoi"),
) -> dict[str, float]:
    """One-stop metric evaluation; returns CPU float scalars."""
    metrics: dict[str, float] = {}
    if "si_sdr" in metric_list:
        metrics["si_sdr"] = _safe_mean(si_sdr_batch(est, ref))
    if "si_sdri" in metric_list:
        metrics["si_sdri"] = _safe_mean(si_sdri_batch(est, ref, mixture))
    if "pesq_wb" in metric_list:
        metrics["pesq_wb"] = _safe_mean(pesq_batch(est, ref, sample_rate, "wb"))
    if "pesq_nb" in metric_list:
        metrics["pesq_nb"] = _safe_mean(pesq_batch(est, ref, sample_rate, "nb"))
    if "stoi" in metric_list:
        metrics["stoi"] = _safe_mean(stoi_batch(est, ref, sample_rate, extended=False))
    if "estoi" in metric_list:
        metrics["estoi"] = _safe_mean(stoi_batch(est, ref, sample_rate, extended=True))
    return metrics
