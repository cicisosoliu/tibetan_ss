"""Multi-scale STFT PatchGAN discriminator (frequency domain).

Operates on ``|STFT(x)|`` at several window sizes and returns per-scale
logits. Used to enforce "purity" on each separated speaker's output.

We implement the multi-scale idea as *multi-resolution STFTs* (common in
GAN vocoders such as HiFi-GAN / SoundStream) combined with a PatchGAN-style
2-D CNN — this matches the spec in the docx ("Multi-scale PatchGAN (频谱域)").
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGANSpec(nn.Module):
    """PatchGAN CNN over a log-magnitude spectrogram tensor of shape (B, 1, F, T)."""

    def __init__(self, channels: tuple[int, ...] = (32, 64, 128, 256, 512)):
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for i, c in enumerate(channels):
            stride = 2 if i < len(channels) - 1 else 1
            layers.append(nn.utils.spectral_norm(
                nn.Conv2d(in_ch, c, kernel_size=3, stride=stride, padding=1)))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = c
        layers.append(nn.Conv2d(in_ch, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.net(spec)


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-resolution PatchGAN discriminator on STFT magnitude spectrograms.

    Parameters
    ----------
    n_ffts : tuple[int, ...]
        FFT window sizes – each spawns an independent PatchGAN sub-net.
    """

    def __init__(self, n_ffts: tuple[int, ...] = (512, 1024, 2048),
                 hop_ratio: float = 0.25,
                 channels: tuple[int, ...] = (32, 64, 128, 256, 512)):
        super().__init__()
        self.n_ffts = tuple(int(n) for n in n_ffts)
        self.hop_ratio = hop_ratio
        self.scales = nn.ModuleList([PatchGANSpec(channels=channels) for _ in self.n_ffts])

    def _compute_spec(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        hop = max(1, int(n_fft * self.hop_ratio))
        win = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
        X = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=win, return_complex=True, center=True, pad_mode="reflect",
        )
        mag = torch.log1p(X.abs())          # log-magnitude for better dynamic range
        return mag.unsqueeze(1)             # (B, 1, F, T')

    def forward(self, audio: torch.Tensor) -> list[torch.Tensor]:
        """Return a list of per-scale logits (one tensor per FFT size)."""
        if audio.ndim == 3:
            audio = audio.reshape(-1, audio.shape[-1])
        return [disc(self._compute_spec(audio, n_fft))
                for n_fft, disc in zip(self.n_ffts, self.scales)]
