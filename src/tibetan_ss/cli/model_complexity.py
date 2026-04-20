"""Report model complexity: params, MACs, inference latency, peak memory.

Usage::

    PYTHONPATH=src python -m tibetan_ss.cli.model_complexity \\
        --config configs/experiment/proposed_formal.yaml \\
        --duration 3.0 --device cuda

Writes ``outputs/logs/<tag>/complexity.json`` with:

    {
        "model": "proposed",
        "params_M": 45.2,
        "macs_G_per_second": 12.3,
        "rtf": 0.012,
        "peak_memory_MB": 420,
        "inference_ms": 36.2
    }
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..models import build_model
from ..utils import set_seed
from ..utils.config import load_config


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_macs(model: torch.nn.Module, x: torch.Tensor) -> float | None:
    """Try to compute MACs via fvcore; return None if not installed."""
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, (x,))
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 2          # FLOPs → MACs (1 MAC ≈ 2 FLOPs)
    except ImportError:
        print("[complexity] fvcore not installed — skipping MACs. pip install fvcore")
        return None
    except Exception as e:
        print(f"[complexity] fvcore failed: {e}")
        return None


def _measure_latency(model: torch.nn.Module, x: torch.Tensor,
                     n_warmup: int = 10, n_runs: int = 50,
                     device: str = "cuda") -> tuple[float, float]:
    """Return (mean_ms, peak_memory_MB)."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_runs * 1000     # ms

        peak_mb = 0.0
        if device == "cuda":
            peak_mb = torch.cuda.max_memory_allocated() / 1e6

    return elapsed, peak_mb


def main() -> None:
    p = argparse.ArgumentParser(description="Model complexity report.")
    p.add_argument("--config", required=True)
    p.add_argument("--duration", type=float, default=3.0, help="Test signal length (seconds)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--output", default=None, help="JSON output path (default: <save_dir>/complexity.json)")
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(42)
    sr = int(cfg["data"]["sample_rate"])
    tag = str(cfg.get("tag", "model"))

    model = build_model({
        **cfg["model"],
        "sample_rate": sr,
        "num_speakers": int(cfg["data"]["mixing"]["num_speakers"]),
    })
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    T = int(args.duration * sr)
    x = torch.randn(1, T, device=device)

    params = _count_params(model)
    macs = _count_macs(model, x)
    latency_ms, peak_mb = _measure_latency(model, x, device=args.device)
    rtf = (latency_ms / 1000) / args.duration       # Real-Time Factor

    result = {
        "model": cfg["model"].get("name", tag),
        "tag": tag,
        "sample_rate": sr,
        "duration_s": args.duration,
        "params_total": params,
        "params_M": round(params / 1e6, 2),
        "macs_total": macs,
        "macs_G_per_second": round(macs / 1e9 / args.duration, 2) if macs else None,
        "inference_ms": round(latency_ms, 2),
        "rtf": round(rtf, 4),
        "peak_memory_MB": round(peak_mb, 1),
        "device": str(device),
    }

    out_path = args.output
    if not out_path:
        save_dir = Path(cfg["training"]["logger"]["save_dir"]) / tag
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(save_dir / "complexity.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*50}")
    print(f" Model: {result['model']}")
    print(f" Params: {result['params_M']:.2f} M")
    if result["macs_G_per_second"]:
        print(f" MACs:   {result['macs_G_per_second']:.2f} G/s")
    print(f" Latency: {result['inference_ms']:.1f} ms ({args.duration}s input)")
    print(f" RTF:    {result['rtf']:.4f}")
    if peak_mb > 0:
        print(f" Peak memory: {result['peak_memory_MB']:.0f} MB")
    print(f" Output: {out_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
