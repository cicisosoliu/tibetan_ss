"""Unified training entry point. Picks the correct Lightning module based on
``engine:`` in the experiment config (``standard`` or ``gan``)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf

from ..data import TibetanMixDataModule
from ..engine import ProposedGANModule, SeparationModule
from ..models import build_model
from ..utils import get_logger, set_seed
from ..utils.config import resolve_defaults as _resolve_defaults_shared


def _build_logger(cfg: dict, save_dir: Path) -> pl.pytorch.loggers.Logger:
    name = cfg["logger"].get("name", "tensorboard")
    if name == "tensorboard":
        return TensorBoardLogger(save_dir=str(save_dir), name="tb", default_hp_metric=False)
    if name == "csv":
        return CSVLogger(save_dir=str(save_dir), name="csv")
    raise ValueError(f"Unknown logger: {name}")


def _build_callbacks(cfg: dict, save_dir: Path) -> list:
    callbacks: list = [LearningRateMonitor(logging_interval="epoch")]
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

    # ---- lightning module --------------------------------------------
    engine_name = str(cfg.get("engine", "standard"))
    if engine_name == "standard":
        pl_module = SeparationModule(
            model=model, training_cfg=cfg["training"],
            sample_rate=int(cfg["data"]["sample_rate"]),
            eval_metrics=tuple(cfg["training"].get("eval_metrics",
                                                    ["si_sdr", "si_sdri", "pesq_wb", "stoi"])),
        )
    elif engine_name == "gan":
        disc_cfg = cfg["model"].get("discriminator", {})
        sched_cfg = cfg["model"].get("schedule", {})
        pl_module = ProposedGANModule(
            model=model, training_cfg=cfg["training"],
            sample_rate=int(cfg["data"]["sample_rate"]),
            discriminator_cfg=disc_cfg, schedule_cfg=sched_cfg,
            eval_metrics=tuple(cfg["training"].get("eval_metrics",
                                                    ["si_sdr", "si_sdri", "pesq_wb", "stoi"])),
        )
    else:
        raise ValueError(f"Unknown engine: {engine_name}")

    # ---- trainer ------------------------------------------------------
    pl_logger = _build_logger(cfg["training"], save_dir)
    callbacks = _build_callbacks(cfg["training"], save_dir)
    tr = cfg["training"]["trainer"]
    trainer = pl.Trainer(
        max_epochs=int(tr["max_epochs"]),
        precision=str(tr.get("precision", "16-mixed")),
        accelerator=str(tr.get("accelerator", "auto")),
        devices=tr.get("devices", "auto"),
        strategy=tr.get("strategy", "auto"),
        gradient_clip_val=float(tr.get("gradient_clip_val", 0.0)) if engine_name == "standard" else 0.0,
        accumulate_grad_batches=int(tr.get("accumulate_grad_batches", 1)),
        log_every_n_steps=int(tr.get("log_every_n_steps", 50)),
        deterministic=bool(tr.get("deterministic", False)),
        logger=pl_logger,
        callbacks=callbacks,
        default_root_dir=str(save_dir),
    )
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
