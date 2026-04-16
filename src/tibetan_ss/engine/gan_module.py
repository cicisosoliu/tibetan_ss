"""Lightning module for the proposed model.

Implements the three-stage training schedule specified in ``提出模型.docx``:

1. **Stage 1** – only ``L_main = PIT + SI-SDR``
2. **Stage 2** – add the representation-divergence loss ``L_rep``
3. **Stage 3** – enable the STFT discriminator (``L_D`` + generator hinge)

The stage boundaries are configured via ``training.schedule`` in the
experiment YAML. Manual optimisation is used because Lightning's dual-
optimizer automation does not play well with conditional generator updates.
"""

from __future__ import annotations

from typing import Any

import lightning as pl
import torch
import torch.nn as nn

from ..losses.pit import pit_si_sdr_loss, reorder_sources
from ..models.proposed.discriminator import MultiScaleSTFTDiscriminator
from ..models.proposed.losses import (
    hinge_discriminator_loss,
    hinge_generator_loss,
    representation_diff_loss,
)
from .metrics import evaluate_batch


class ProposedGANModule(pl.LightningModule):
    """Trainer for :class:`ProposedEarlySeparation` with auxiliary losses."""

    def __init__(self, model: nn.Module, training_cfg: dict, sample_rate: int = 16000,
                 discriminator_cfg: dict | None = None,
                 schedule_cfg: dict | None = None,
                 eval_metrics: tuple[str, ...] = ("si_sdr", "si_sdri", "pesq_wb", "stoi")):
        super().__init__()
        self.model = model
        self.discriminator = MultiScaleSTFTDiscriminator(**(discriminator_cfg or {}))
        self.training_cfg = training_cfg
        self.sample_rate = sample_rate
        self.eval_metrics = tuple(eval_metrics)
        self.schedule_cfg = schedule_cfg or {"rep_from_epoch": 10, "gan_from_epoch": 25}
        # The proposed trainer does its own optimiser juggling.
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["model"])

    # ------------------------------------------------------------------
    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        return self.model(mixture)

    # ------------------------------------------------------------------
    @property
    def _rep_enabled(self) -> bool:
        return self.current_epoch >= int(self.schedule_cfg.get("rep_from_epoch", 0))

    @property
    def _gan_enabled(self) -> bool:
        return self.current_epoch >= int(self.schedule_cfg.get("gan_from_epoch", 10**9))

    # ------------------------------------------------------------------
    def training_step(self, batch: dict, batch_idx: int):
        opt_g, opt_d = self.optimizers()
        if self._gan_enabled:
            # ----- Step 1: update discriminator on detached fakes -------
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                est, _ = self.model(batch["mixture"], return_aux=True)
            # Align order for a meaningful D target
            _, perm = pit_si_sdr_loss(est, batch["sources"], return_perm=True)
            est_aligned = reorder_sources(est, perm)
            real = batch["sources"].reshape(-1, batch["sources"].shape[-1])
            fake = est_aligned.reshape(-1, est_aligned.shape[-1]).detach()
            d_real = self.discriminator(real)
            d_fake = self.discriminator(fake)
            loss_d = hinge_discriminator_loss(d_real, d_fake)
            self.manual_backward(loss_d)
            opt_d.step()
            self.log("train/loss_d", loss_d, on_step=True, on_epoch=True, prog_bar=True,
                     batch_size=batch["mixture"].shape[0])

        # ----- Step 2: update separator --------------------------------
        opt_g.zero_grad(set_to_none=True)
        est, aux = self.model(batch["mixture"], return_aux=True)
        loss_main, perm = pit_si_sdr_loss(est, batch["sources"], return_perm=True)
        est_aligned = reorder_sources(est, perm)
        total = loss_main
        self.log("train/loss_main", loss_main, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=batch["mixture"].shape[0])

        if self._rep_enabled:
            lam_rep = float(self.schedule_cfg.get("rep_weight", 0.05))
            loss_rep = representation_diff_loss(aux["z_a"], aux["z_b"])
            total = total + lam_rep * loss_rep
            self.log("train/loss_rep", loss_rep, on_step=False, on_epoch=True,
                     batch_size=batch["mixture"].shape[0])

        if self._gan_enabled:
            lam_g = float(self.schedule_cfg.get("gan_weight", 0.1))
            d_fake_for_g = self.discriminator(est_aligned.reshape(-1, est_aligned.shape[-1]))
            loss_g = hinge_generator_loss(d_fake_for_g)
            total = total + lam_g * loss_g
            self.log("train/loss_g", loss_g, on_step=False, on_epoch=True,
                     batch_size=batch["mixture"].shape[0])

        self.manual_backward(total)
        clip = float(self.training_cfg["trainer"].get("gradient_clip_val", 0.0))
        if clip > 0:
            # Use torch directly (not self.clip_gradients) because the Trainer
            # is configured *without* gradient_clip_val when this module is
            # running — passing both would raise a MisconfigurationException.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
        opt_g.step()
        self.log("train/loss", total, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=batch["mixture"].shape[0])
        return total

    # ------------------------------------------------------------------
    def _eval_step(self, batch: dict, stage: str) -> None:
        mix = batch["mixture"]
        ref = batch["sources"]
        with torch.no_grad():
            est = self.model(mix)
            loss, perm = pit_si_sdr_loss(est, ref, return_perm=True)
            est_aligned = reorder_sources(est, perm)
            metrics = evaluate_batch(est_aligned, ref, mix, self.sample_rate, self.eval_metrics)
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True,
                 prog_bar=(stage == "val"), batch_size=mix.shape[0])
        for k, v in metrics.items():
            self.log(f"{stage}/{k}", v, on_step=False, on_epoch=True,
                     prog_bar=(k == "si_sdri"), batch_size=mix.shape[0])

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        self._eval_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        self._eval_step(batch, "test")

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt_cfg = self.training_cfg["optimizer"]
        lr = float(opt_cfg["lr"])
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        wd = float(opt_cfg.get("weight_decay", 0.0))
        # Separator (generator) — the main model
        opt_g = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=wd)
        # Discriminator gets a lower LR by default
        d_lr = float(self.training_cfg.get("disc_lr", lr * 0.5))
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), lr=d_lr, betas=betas, weight_decay=wd)
        return [opt_g, opt_d]
