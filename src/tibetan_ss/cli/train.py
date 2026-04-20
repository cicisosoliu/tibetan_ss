"""Unified training entry point. Picks the correct Lightning module based on
``engine:`` in the experiment config (``standard`` or ``gan``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf

from ..data import TibetanMixDataModule
from ..engine import ProposedGANModule, SeparationModule
from ..models import build_model
from ..utils import get_logger, set_seed
from ..utils.config import resolve_defaults as _resolve_defaults_shared


class DynamicMixingEpochCallback(pl.Callback):
    """Bump the train dataset's epoch counter at the start of each epoch.

    ``LightningDataModule`` does NOT have ``on_train_epoch_start``, so this
    must live in a ``Callback`` (which Lightning *does* call). The dataset
    stores its epoch in a ``multiprocessing.Value`` so all DataLoader workers
    (including ``persistent_workers``) see the update.
    """

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        dm = trainer.datamodule
        if dm is not None and hasattr(dm, "_train") and dm._train is not None:
            ds = dm._train
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(trainer.current_epoch)


class TrainingETACallback(pl.Callback):
    """Print real-time ETA after each training epoch.

    Shows: current epoch, time/epoch, ETA for this model, and (optionally)
    ETA for the entire multi-model pipeline.
    """

    def __init__(self, tag: str = "", total_models: int = 1, models_done: int = 0):
        super().__init__()
        self.tag = tag
        self.total_models = total_models
        self.models_done = models_done
        self._fit_start: float = 0.0
        self._epoch_start: float = 0.0

    @staticmethod
    def _fmt(secs: float) -> str:
        secs = max(0, int(secs))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h{m:02d}m{s:02d}s"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def on_fit_start(self, trainer, pl_module):
        import time
        self._fit_start = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        import time
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        import time
        now = time.time()
        epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs
        elapsed = now - self._fit_start
        epoch_time = now - self._epoch_start
        epochs_done = epoch + 1

        # Average time per epoch (smoothed over all completed epochs)
        avg_per_epoch = elapsed / max(epochs_done, 1)
        remaining_epochs = max_epochs - epochs_done
        eta_model = avg_per_epoch * remaining_epochs

        # Metrics from the current epoch (if logged)
        metrics_str = ""
        for key in ("val/si_sdri", "val/loss", "train/loss"):
            val = trainer.callback_metrics.get(key)
            if val is not None:
                metrics_str += f"  {key}={float(val):.3f}"

        # ETA for remaining models in the pipeline
        eta_pipeline_str = ""
        if self.total_models > 1 and self.models_done < self.total_models:
            # Estimate: current model remaining + avg_model_time × remaining_models
            models_left_after = self.total_models - self.models_done - 1
            # Use current model's avg_per_epoch × max_epochs as estimate for other models
            est_per_model = avg_per_epoch * max_epochs
            eta_pipeline = eta_model + est_per_model * models_left_after
            eta_pipeline_str = f"  │  Pipeline ETA: ~{self._fmt(eta_pipeline)} ({self.models_done+1}/{self.total_models})"

        print(
            f"\n  ⏱  [{self.tag}] Epoch {epochs_done}/{max_epochs}"
            f"  │  This epoch: {self._fmt(epoch_time)}"
            f"  │  Avg: {self._fmt(avg_per_epoch)}/epoch"
            f"  │  Model ETA: ~{self._fmt(eta_model)}"
            f"{eta_pipeline_str}"
            f"\n     {metrics_str}",
            flush=True,
        )


def _build_logger(cfg: dict, save_dir: Path) -> pl.pytorch.loggers.Logger:
    name = cfg["logger"].get("name", "tensorboard")
    if name == "tensorboard":
        return TensorBoardLogger(save_dir=str(save_dir), name="tb", default_hp_metric=False)
    if name == "csv":
        return CSVLogger(save_dir=str(save_dir), name="csv")
    raise ValueError(f"Unknown logger: {name}")


def _build_callbacks(cfg: dict, save_dir: Path, tag: str = "",
                     total_models: int = 1, models_done: int = 0) -> list:
    callbacks: list = [
        LearningRateMonitor(logging_interval="epoch"),
        DynamicMixingEpochCallback(),         # bumps DM epoch via mp.Value
        TrainingETACallback(tag=tag, total_models=total_models,
                            models_done=models_done),
    ]
    ckpt_cfg = cfg.get("checkpoint", {})
    callbacks.append(ModelCheckpoint(
        dirpath=str(save_dir / "checkpoints"),
        monitor=ckpt_cfg.get("monitor", "val/si_sdri"),
        mode=ckpt_cfg.get("mode", "max"),
        save_top_k=int(ckpt_cfg.get("save_top_k", 3)),
        save_last=bool(ckpt_cfg.get("save_last", True)),
        filename="epoch{epoch:03d}-sisdri{val/si_sdri:.3f}",
        auto_insert_metric_name=False,
    ))
    es = cfg.get("early_stop", {})
    if es.get("enabled", False):
        callbacks.append(EarlyStopping(
            monitor=es.get("monitor", "val/si_sdri"),
            mode=es.get("mode", "max"),
            patience=int(es.get("patience", 20)),
        ))
    return callbacks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to experiment YAML")
    p.add_argument("overrides", nargs="*", help="Hydra-style overrides (dot.key=value)")
    args = p.parse_args()

    base = OmegaConf.load(args.config)
    overrides = OmegaConf.from_dotlist(args.overrides)
    cfg = OmegaConf.merge(_resolve_defaults(base, Path(args.config)), overrides)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    logger = get_logger("train")

    tag = str(cfg.get("tag", "run"))
    save_dir = Path(cfg["training"]["logger"]["save_dir"]).expanduser() / tag
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "resolved.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(config=OmegaConf.create(cfg), f=f)
    logger.info(f"[train] tag={tag} save_dir={save_dir}")

    # ---- data ---------------------------------------------------------
    datamodule = TibetanMixDataModule(cfg=cfg["data"], training_cfg=cfg["training"])

    # ---- model --------------------------------------------------------
    model = build_model({
        **cfg["model"],
        "sample_rate": int(cfg["data"]["sample_rate"]),
        "num_speakers": int(cfg["data"]["mixing"]["num_speakers"]),
    })

    # Optional: torch.compile for 20-40% speedup on H100/A100 (PyTorch 2.x)
    if cfg.get("compile", False):
        logger.info("[train] applying torch.compile (first few steps will be slow due to tracing)")
        model = torch.compile(model)

    # ---- lightning module --------------------------------------------
    engine_name = str(cfg.get("engine", "standard"))
    _eval_metrics = tuple(cfg["training"].get("eval_metrics",
                                               ["si_sdr", "si_sdri", "pesq_wb", "stoi"]))
    _test_max_audio = int(cfg.get("test_max_audio", 50))
    if engine_name == "standard":
        pl_module = SeparationModule(
            model=model, training_cfg=cfg["training"],
            sample_rate=int(cfg["data"]["sample_rate"]),
            eval_metrics=_eval_metrics,
            test_save_dir=str(save_dir),
            test_max_audio=_test_max_audio,
        )
    elif engine_name == "gan":
        disc_cfg = cfg["model"].get("discriminator", {})
        sched_cfg = cfg["model"].get("schedule", {})
        # `disc_lr` is declared at the model level in configs/model/proposed.yaml
        # but consumed by ProposedGANModule.configure_optimizers via training_cfg,
        # so surface it here.
        training_cfg = dict(cfg["training"])
        if "disc_lr" in cfg["model"]:
            training_cfg["disc_lr"] = cfg["model"]["disc_lr"]
        pl_module = ProposedGANModule(
            model=model, training_cfg=training_cfg,
            sample_rate=int(cfg["data"]["sample_rate"]),
            discriminator_cfg=disc_cfg, schedule_cfg=sched_cfg,
            eval_metrics=_eval_metrics,
            test_save_dir=str(save_dir),
            test_max_audio=_test_max_audio,
        )
    else:
        raise ValueError(f"Unknown engine: {engine_name}")

    # ---- trainer ------------------------------------------------------
    pl_logger = _build_logger(cfg["training"], save_dir)
    callbacks = _build_callbacks(
        cfg["training"], save_dir, tag=tag,
        total_models=int(cfg.get("_total_models", 1)),
        models_done=int(cfg.get("_models_done", 0)),
    )
    tr = cfg["training"]["trainer"]
    trainer_kwargs = dict(
        max_epochs=int(tr["max_epochs"]),
        precision=str(tr.get("precision", "16-mixed")),
        accelerator=str(tr.get("accelerator", "auto")),
        devices=tr.get("devices", "auto"),
        strategy=tr.get("strategy", "auto"),
        accumulate_grad_batches=int(tr.get("accumulate_grad_batches", 1)),
        check_val_every_n_epoch=int(tr.get("check_val_every_n_epoch", 1)),
        log_every_n_steps=int(tr.get("log_every_n_steps", 50)),
        deterministic=bool(tr.get("deterministic", False)),
        logger=pl_logger,
        callbacks=callbacks,
        default_root_dir=str(save_dir),
    )
    # Lightning's auto-clipping is only engaged for the ``standard`` engine.
    # The GAN engine uses manual optimization and handles clipping inside
    # ``ProposedGANModule.training_step`` — passing gradient_clip_val here
    # would conflict with that path (MisconfigurationException).
    if engine_name == "standard":
        trainer_kwargs["gradient_clip_val"] = float(tr.get("gradient_clip_val", 0.0))
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(pl_module, datamodule=datamodule)
    trainer.test(pl_module, datamodule=datamodule, ckpt_path="best")


# ---------------------------------------------------------------------------
# Minimal hydra-compat: resolve ``defaults: [ { key: value }, ... ]`` against
# the sibling configs directory. We do this manually so we don't require the
# full Hydra plugin machinery on the user's machine.
# ---------------------------------------------------------------------------

def _resolve_defaults(cfg, cfg_path: Path):
    """Hydra-lite defaults resolver (shared with data-prep scripts)."""
    return _resolve_defaults_shared(cfg, cfg_path)


if __name__ == "__main__":
    main()
