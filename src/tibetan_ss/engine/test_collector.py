"""Mixin for collecting per-utterance test results + saving separated audio.

Both ``SeparationModule`` and ``ProposedGANModule`` inherit this so the test
pipeline is identical regardless of training engine.

After ``trainer.test()`` completes, the following files are written under
``<save_dir>/test_results/``:

* ``per_utterance.csv`` — one row per test sample with columns:
    id, si_sdr, si_sdri, pesq_wb, stoi, overlap_ratio,
    effective_overlap_ratio, snr_db, gender_pair, level_diff_db
* ``summary.json`` — overall means + medians + std
* ``audio/<sample_id>/{mixture,s1_est,s2_est,s1_ref,s2_ref}.wav``
    (first ``max_audio_save`` samples)
* ``features/<sample_id>.pt`` — intermediate representations z_a/z_b
    from the proposed model (for t-SNE; only when ``return_aux`` available)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from ..losses.pit import pit_si_sdr_loss, reorder_sources
from ..losses.sisdr import si_sdr
from .metrics import pesq_batch, stoi_batch, si_sdr_batch, si_sdri_batch, _safe_mean


class TestCollectorMixin:
    """Mixin — add to a LightningModule to collect per-utterance test data."""

    # Set these in __init__ of the concrete class:
    #   self._test_results: list[dict] = []
    #   self._test_save_dir: Path | None = None
    #   self._test_max_audio: int = 50
    #   self._test_save_features: bool = False

    def _init_test_collector(self, save_dir: str | Path | None = None,
                             max_audio_save: int = 50,
                             save_features: bool = False) -> None:
        self._test_results: list[dict] = []
        self._test_save_dir = Path(save_dir) if save_dir else None
        self._test_max_audio = int(max_audio_save)
        self._test_save_features = save_features
        self._test_audio_count = 0

    # ------------------------------------------------------------------
    def _collect_test_step(
        self,
        batch: dict,
        est_aligned: torch.Tensor,
        ref: torch.Tensor,
        mix: torch.Tensor,
        sample_rate: int,
        eval_metrics: tuple[str, ...],
        aux: dict | None = None,
    ) -> None:
        """Called from ``test_step`` after PIT alignment."""
        B = mix.shape[0]
        ids = batch.get("id", [str(i) for i in range(B)])
        metas = batch.get("meta", [{}] * B)
        if not isinstance(metas, list):
            metas = [{}] * B
        lengths = batch.get("length", None)

        # Per-utterance metrics (no batch averaging)
        with torch.no_grad():
            sdr_vals = si_sdr_batch(est_aligned, ref)               # (B, C)
            sdri_vals = si_sdri_batch(est_aligned, ref, mix)        # (B, C)
            pesq_vals = pesq_batch(est_aligned, ref, sample_rate, "wb") if "pesq_wb" in eval_metrics else None
            stoi_vals = stoi_batch(est_aligned, ref, sample_rate) if "stoi" in eval_metrics else None

        for i in range(B):
            T = int(lengths[i]) if lengths is not None else mix.shape[-1]
            meta = metas[i] if i < len(metas) else {}
            row = {
                "id": ids[i] if i < len(ids) else f"unk_{i}",
                "si_sdr": float(sdr_vals[i].mean().item()),
                "si_sdri": float(sdri_vals[i].mean().item()),
                "pesq_wb": float(pesq_vals[i].mean().item()) if pesq_vals is not None else float("nan"),
                "stoi": float(stoi_vals[i].mean().item()) if stoi_vals is not None else float("nan"),
                "overlap_ratio": meta.get("overlap_ratio", float("nan")),
                "effective_overlap_ratio": meta.get("effective_overlap_ratio", float("nan")),
                "snr_db": meta.get("snr_db") if meta.get("snr_db") is not None else float("nan"),
                "gender_pair": meta.get("gender_pair", ""),
                "level_diff_db": meta.get("level_diff_db", float("nan")),
                "length_samples": T,
            }
            self._test_results.append(row)

            # Save audio for a subset of samples
            if self._test_save_dir and self._test_audio_count < self._test_max_audio:
                self._save_audio(ids[i] if i < len(ids) else f"sample_{len(self._test_results)}",
                                 mix[i, :T], est_aligned[i, :, :T], ref[i, :, :T],
                                 sample_rate)
                self._test_audio_count += 1

            # Save intermediate features for t-SNE (proposed model only)
            if self._test_save_features and aux is not None and self._test_save_dir:
                feat_dir = self._test_save_dir / "test_results" / "features"
                feat_dir.mkdir(parents=True, exist_ok=True)
                sample_id = ids[i] if i < len(ids) else f"sample_{len(self._test_results)}"
                feat_path = feat_dir / f"{sample_id}.pt"
                feats_to_save = {}
                for key in ("z_a", "z_b", "features_a", "features_b"):
                    if key in aux:
                        feats_to_save[key] = aux[key][i].detach().cpu()
                if feats_to_save:
                    torch.save(feats_to_save, feat_path)

    # ------------------------------------------------------------------
    def _save_audio(self, sample_id: str, mix: torch.Tensor,
                    est: torch.Tensor, ref: torch.Tensor,
                    sample_rate: int) -> None:
        audio_dir = self._test_save_dir / "test_results" / "audio" / sample_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(audio_dir / "mixture.wav"),
                        mix.unsqueeze(0).float().cpu(), sample_rate)
        for k in range(est.shape[0]):
            torchaudio.save(str(audio_dir / f"s{k+1}_est.wav"),
                            est[k:k+1].float().cpu(), sample_rate)
            torchaudio.save(str(audio_dir / f"s{k+1}_ref.wav"),
                            ref[k:k+1].float().cpu(), sample_rate)

    # ------------------------------------------------------------------
    def _finalize_test(self) -> None:
        """Write per_utterance.csv + summary.json after all test steps."""
        if not self._test_results or not self._test_save_dir:
            return
        out_dir = self._test_save_dir / "test_results"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Per-utterance CSV
        csv_path = out_dir / "per_utterance.csv"
        fieldnames = list(self._test_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._test_results)

        # Summary JSON
        numeric_keys = ["si_sdr", "si_sdri", "pesq_wb", "stoi"]
        summary: dict[str, Any] = {"n_samples": len(self._test_results)}
        for k in numeric_keys:
            vals = [r[k] for r in self._test_results if np.isfinite(r[k])]
            if vals:
                arr = np.array(vals)
                summary[k] = {
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
        # Per-condition breakdown
        for group_key in ("gender_pair", "overlap_bin", "snr_bin"):
            breakdown = {}
            for r in self._test_results:
                if group_key == "overlap_bin":
                    ov = r.get("overlap_ratio", float("nan"))
                    if np.isnan(ov):
                        g = "unknown"
                    elif ov < 0.3:
                        g = "low(0-0.3)"
                    elif ov < 0.7:
                        g = "mid(0.3-0.7)"
                    else:
                        g = "high(0.7-1.0)"
                elif group_key == "snr_bin":
                    snr = r.get("snr_db", float("nan"))
                    if np.isnan(snr):
                        g = "no_noise"
                    elif snr < 10:
                        g = "low(<10dB)"
                    elif snr < 20:
                        g = "mid(10-20dB)"
                    else:
                        g = "high(>20dB)"
                else:
                    g = r.get(group_key, "unknown") or "unknown"
                breakdown.setdefault(g, []).append(r.get("si_sdri", float("nan")))
            summary[f"si_sdri_by_{group_key}"] = {
                g: {"mean": float(np.nanmean(v)), "n": len(v)}
                for g, v in sorted(breakdown.items())
            }

        summary_path = out_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"[test] wrote {csv_path} ({len(self._test_results)} rows)")
        print(f"[test] wrote {summary_path}")
        if self._test_audio_count > 0:
            print(f"[test] saved {self._test_audio_count} audio samples to {out_dir / 'audio'}")
