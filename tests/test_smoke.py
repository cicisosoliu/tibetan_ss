"""Smoke tests — don't require heavyweight deps (no mamba-ssm / speechbrain)."""

from __future__ import annotations

import numpy as np
import torch


def test_mixture_simulator_shapes() -> None:
    from tibetan_ss.data.mixing import MixingConfig, MixtureSimulator
    cfg = MixingConfig(sample_rate=16000, segment_seconds=2.0, random_length=False)
    sim = MixtureSimulator(cfg)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(16000 * 3).astype(np.float32)
    b = rng.standard_normal(16000 * 3).astype(np.float32)
    n = rng.standard_normal(16000 * 4).astype(np.float32) * 0.1
    out = sim.simulate(a, b, n, rng=rng, gender_a="M", gender_b="F")
    assert out.mixture.shape == (32000,)
    assert out.sources.shape == (2, 32000)
    assert out.meta["gender_pair"] in ("FM",)


def test_sisdr_basic() -> None:
    from tibetan_ss.losses.sisdr import si_sdr
    ref = torch.randn(2, 16000)
    assert torch.isfinite(si_sdr(ref, ref)).all()
    noisy = ref + 0.1 * torch.randn_like(ref)
    assert (si_sdr(noisy, ref) > 0).all()


def test_pit_loss_permutation() -> None:
    from tibetan_ss.losses.pit import pit_si_sdr_loss
    ref = torch.randn(4, 2, 16000)
    est = ref[:, [1, 0], :]                         # swapped
    loss, perm = pit_si_sdr_loss(est, ref, return_perm=True)
    assert loss.item() < 0                           # good SI-SDR → negative loss
    assert (perm[:, 0] == 1).all() and (perm[:, 1] == 0).all()


def test_identity_separator_output_shape() -> None:
    from tibetan_ss.models import build_model
    model = build_model({"name": "identity", "sample_rate": 16000, "num_speakers": 2})
    x = torch.randn(2, 16000)
    y = model(x)
    assert y.shape == (2, 2, 16000)


def test_proposed_model_forward() -> None:
    from tibetan_ss.models import build_model
    model = build_model({
        "name": "proposed", "sample_rate": 16000, "num_speakers": 2,
        "n_filters": 64, "kernel_size": 32, "bottleneck": 32, "tcn_hidden": 64,
        "encoder_tcn_blocks": 2, "encoder_tcn_repeats": 1,
        "branch_tcn_blocks": 2, "branch_tcn_repeats": 1,
        "decoder_tcn_blocks": 2, "decoder_tcn_repeats": 1,
    })
    x = torch.randn(2, 16000)
    y = model(x)
    assert y.shape == (2, 2, 16000)
    y2, aux = model(x, return_aux=True)
    assert y2.shape == (2, 2, 16000)
    assert "z_a" in aux and "z_b" in aux


def test_discriminator_output_list() -> None:
    from tibetan_ss.models.proposed.discriminator import MultiScaleSTFTDiscriminator
    d = MultiScaleSTFTDiscriminator(n_ffts=(512, 1024))
    logits = d(torch.randn(2, 16000))
    assert isinstance(logits, list) and len(logits) == 2
    for l in logits:
        assert l.ndim == 4                         # (B, 1, F, T)


def test_registry_contains_all_models() -> None:
    from tibetan_ss.models import list_models
    names = set(list_models())
    assert {"identity", "proposed", "dip_frontend"}.issubset(names)
