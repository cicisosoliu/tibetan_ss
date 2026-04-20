"""Visualization utilities for paper figures.

Three modes:

1. **spectrogram** — Plot mixture / estimated / reference spectrograms side by side
2. **curves** — Plot training curves (SI-SDRi vs epoch) for all models
3. **tsne** — t-SNE of dual-branch representations z_a/z_b (proposed model only)

Usage::

    # Spectrograms
    python -m tibetan_ss.cli.visualize spectrogram \\
        --audio-dir outputs/logs/proposed_formal/test_results/audio/sample_0 \\
        --output figures/spectrogram_sample0.pdf

    # Training curves from TensorBoard logs
    python -m tibetan_ss.cli.visualize curves \\
        --root outputs/logs --metric val/si_sdri \\
        --output figures/training_curves.pdf

    # t-SNE of branch representations
    python -m tibetan_ss.cli.visualize tsne \\
        --features-dir outputs/logs/proposed_formal/test_results/features \\
        --output figures/tsne_branches.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _spectrogram(args) -> None:
    import matplotlib.pyplot as plt
    import soundfile as sf

    audio_dir = Path(args.audio_dir)
    files = {
        "Mixture": audio_dir / "mixture.wav",
        "Estimated S1": audio_dir / "s1_est.wav",
        "Reference S1": audio_dir / "s1_ref.wav",
        "Estimated S2": audio_dir / "s2_est.wav",
        "Reference S2": audio_dir / "s2_ref.wav",
    }

    fig, axes = plt.subplots(len(files), 1, figsize=(10, 2.2 * len(files)),
                              sharex=True, constrained_layout=True)
    for ax, (label, path) in zip(axes, files.items()):
        if not path.exists():
            ax.set_title(f"{label} (not found)")
            continue
        wav, sr = sf.read(str(path))
        ax.specgram(wav, NFFT=512, Fs=sr, noverlap=384, cmap="magma")
        ax.set_ylabel("Hz")
        ax.set_title(label, fontsize=10)
    axes[-1].set_xlabel("Time (s)")
    fig.savefig(args.output, dpi=200)
    print(f"[vis] wrote {args.output}")
    plt.close(fig)


def _curves(args) -> None:
    """Read TensorBoard event files and plot training curves."""
    import matplotlib.pyplot as plt

    root = Path(args.root)
    metric = args.metric

    fig, ax = plt.subplots(figsize=(8, 5))
    for run_dir in sorted(root.glob("*")):
        tb_dir = run_dir / "tb"
        if not tb_dir.exists():
            continue
        versions = sorted(tb_dir.glob("version_*"))
        if not versions:
            continue
        # Try to read from CSV first (simpler and no tensorflow dependency)
        csv_dir = run_dir / "csv"
        if csv_dir.exists():
            csv_versions = sorted(csv_dir.glob("version_*"))
            if csv_versions:
                csv_path = csv_versions[-1] / "metrics.csv"
                if csv_path.exists():
                    import csv
                    epochs, vals = [], []
                    with open(csv_path, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if metric in row and row[metric]:
                                epoch_key = "epoch"
                                if epoch_key in row and row[epoch_key]:
                                    epochs.append(int(row[epoch_key]))
                                    vals.append(float(row[metric]))
                    if epochs:
                        ax.plot(epochs, vals, label=run_dir.name, linewidth=1.5)
                        continue

        # Fallback: try tensorboard
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(str(versions[-1]))
            ea.Reload()
            if metric in ea.Tags().get("scalars", []):
                events = ea.Scalars(metric)
                steps = [e.step for e in events]
                vals = [e.value for e in events]
                ax.plot(steps, vals, label=run_dir.name, linewidth=1.5)
        except ImportError:
            pass

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title("Training Curves")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"[vis] wrote {args.output}")
    plt.close(fig)


def _tsne(args) -> None:
    """t-SNE of z_a / z_b from the proposed model's test features."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import torch

    feat_dir = Path(args.features_dir)
    feat_files = sorted(feat_dir.glob("*.pt"))
    if not feat_files:
        print(f"[vis] no .pt files in {feat_dir}")
        return

    za_list, zb_list = [], []
    max_samples = int(args.max_samples)
    for fp in feat_files[:max_samples]:
        d = torch.load(fp, map_location="cpu", weights_only=False)
        if "z_a" in d and "z_b" in d:
            # Global average pool over time: (C, L) → (C,)
            za_list.append(d["z_a"].mean(dim=-1).numpy())
            zb_list.append(d["z_b"].mean(dim=-1).numpy())

    if not za_list:
        print("[vis] no z_a/z_b features found")
        return

    za = np.stack(za_list)                 # (N, C)
    zb = np.stack(zb_list)
    combined = np.concatenate([za, zb], axis=0)
    labels = ["Branch A"] * len(za) + ["Branch B"] * len(zb)

    tsne = TSNE(n_components=2, perplexity=min(30, len(combined) - 1),
                random_state=42)
    emb = tsne.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(7, 6))
    n = len(za)
    ax.scatter(emb[:n, 0], emb[:n, 1], c="tab:blue", alpha=0.6, s=15, label="Branch A (z_a)")
    ax.scatter(emb[n:, 0], emb[n:, 1], c="tab:orange", alpha=0.6, s=15, label="Branch B (z_b)")
    ax.legend(fontsize=10)
    ax.set_title("t-SNE of Dual-Branch Representations")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"[vis] wrote {args.output}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualization for paper figures.")
    sub = parser.add_subparsers(dest="mode", required=True)

    # spectrogram
    sp = sub.add_parser("spectrogram", help="Plot mixture/est/ref spectrograms")
    sp.add_argument("--audio-dir", required=True)
    sp.add_argument("--output", default="figures/spectrogram.pdf")

    # curves
    cp = sub.add_parser("curves", help="Plot training curves for all models")
    cp.add_argument("--root", default="outputs/logs")
    cp.add_argument("--metric", default="val/si_sdri")
    cp.add_argument("--output", default="figures/training_curves.pdf")

    # tsne
    tp = sub.add_parser("tsne", help="t-SNE of dual-branch representations")
    tp.add_argument("--features-dir", required=True)
    tp.add_argument("--max-samples", default=200)
    tp.add_argument("--output", default="figures/tsne_branches.pdf")

    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "spectrogram":
        _spectrogram(args)
    elif args.mode == "curves":
        _curves(args)
    elif args.mode == "tsne":
        _tsne(args)


if __name__ == "__main__":
    main()
