"""Generic Lightning module used by every *discriminative* baseline.

The proposed model, which includes a GAN discriminator, is trained by
:class:`.gan_module.ProposedGANModule` instead.
"""

from __future__ import annotations

from typing import Any

import lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from ..losses.pit import pit_si_sdr_loss, reorder_sources
from .metrics import evaluate_batch


class SeparationModule(pl.LightningModule):
    """Base Lightning module for any 2-speaker separator.

    The wrapped ``model`` must satisfy
    :class:`tibetan_ss.models.base.BaseSeparator` — i.e. accept a mixture of
    shape ``(B, T)`` and return ``(B, 2, T)`` estimates.
    """

    def __init__(self, model: nn.Module, training_cfg: dict, sample_rate: int = 16000,
                 eval_metrics: tuple[str, ...] = ("si_sdr", "si_sdri", "pesq_wb", "stoi")):
        super().__init__()
        self.model = model
        self.training_cfg = training_cfg
        self.sample_rate = sample_rate
        self.eval_metrics = tuple(eval_metrics)
        self.save_hyperparameters(ignore=["model"])

    # ------------------------------------------------------------------
    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        return self.model(mixture)

    # ------------------------------------------------------------------
    def _step(self, batch: dict, stage: str) -> dict[str, Any]:
        mix = batch["mixture"]
        ref = batch["sources"]
        est = self.model(mix)
        loss, perm = pit_si_sdr_loss(est, ref, return_perm=True)
        est_aligned = reorder_sources(est, perm)
        self.log(f"{stage}/loss", loss, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True, batch_size=mix.shape[0])
        return {"loss": loss, "est": est_aligned, "ref": ref, "mix": mix}

    # ------------------------------------------------------------------
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self._step(batch, "train")
        return out["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        out = self._step(batch, "val")
        metrics = evaluate_batch(out["est"], out["ref"], out["mix"], self.sample_rate,
                                 self.eval_metrics)
        for k, v in metrics.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True,
                     prog_bar=(k in ("si_sdri", "si_sdr")),
                     batch_size=out["mix"].shape[0])

    def test_step(self, batch: dict, batch_idx: int) -> None:
        out = self._step(batch, "test")
        metrics = evaluate_batch(out["est"], out["ref"], out["mix"], self.sample_rate,
                                 self.eval_metrics)
        for k, v in metrics.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True,
                     batch_size=out["mix"].shape[0])

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt_cfg = self.training_cfg["optimizer"]
        # Hydra-style _target_ instantiation kept simple to avoid extra deps
        lr = float(opt_cfg["lr"])
        betas = tuple(opt_cfg.get("betas", (0.9, 0.999)))
        wd = float(opt_cfg.get("weight_decay", 0.0))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=wd)

        sched_cfg = self.training_cfg.get("scheduler")
        if sched_cfg is None or sched_cfg.get("name") == "none":
            return optimizer

        if sched_cfg["name"] == "cosine":
            warmup_epochs = int(sched_cfg.get("warmup_epochs", 0))
            min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))
            total_epochs = int(self.training_cfg["trainer"]["max_epochs"])
            if warmup_epochs > 0:
                warmup = LambdaLR(optimizer,
                                  lr_lambda=lambda e: min((e + 1) / warmup_epochs, 1.0))
                cosine = CosineAnnealingLR(optimizer,
                                           T_max=max(1, total_epochs - warmup_epochs),
                                           eta_min=lr * min_lr_ratio)
                scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                         milestones=[warmup_epochs])
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs,
                                              eta_min=lr * min_lr_ratio)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        raise ValueError(f"Unknown scheduler name: {sched_cfg['name']}")
