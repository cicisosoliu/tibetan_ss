"""Evaluate a saved checkpoint on the test split and dump per-split metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as pl
from omegaconf import OmegaConf

from ..data import TibetanMixDataModule
from ..engine import ProposedGANModule, SeparationModule
from ..models import build_model
from ..utils import get_logger, set_seed
from .train import _resolve_defaults


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="experiment YAML used at train time")
    p.add_argument("--checkpoint", required=True, help="Lightning checkpoint (.ckpt)")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--output", default=None, help="optional JSON output path")
    p.add_argument("--save-dir", default=None,
                    help="Override test_save_dir (where per_utterance.csv + audio are written)")
    p.add_argument("--max-audio", type=int, default=50,
                    help="Max number of test audio samples to save")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    base = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(_resolve_defaults(base, Path(args.config)),
                          OmegaConf.from_dotlist(args.overrides))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    set_seed(int(cfg.get("seed", 42)))
    logger = get_logger("evaluate")

    datamodule = TibetanMixDataModule(cfg=cfg["data"], training_cfg=cfg["training"])
    datamodule.setup(stage="test")

    model = build_model({
        **cfg["model"],
        "sample_rate": int(cfg["data"]["sample_rate"]),
        "num_speakers": int(cfg["data"]["mixing"]["num_speakers"]),
    })

    engine_name = str(cfg.get("engine", "standard"))
    _eval_metrics = tuple(cfg["training"].get("eval_metrics",
                                               ["si_sdr", "si_sdri", "pesq_wb", "stoi"]))

    # Determine save_dir: explicit --save-dir > config-derived default
    if args.save_dir:
        save_dir = args.save_dir
    else:
        tag = str(cfg.get("tag", "run"))
        save_dir = str(Path(cfg["training"]["logger"]["save_dir"]).expanduser() / tag)

    if engine_name == "gan":
        disc_cfg = cfg["model"].get("discriminator", {})
        sched_cfg = cfg["model"].get("schedule", {})
        training_cfg = dict(cfg["training"])
        if "disc_lr" in cfg["model"]:
            training_cfg["disc_lr"] = cfg["model"]["disc_lr"]
        pl_module = ProposedGANModule.load_from_checkpoint(
            args.checkpoint, model=model,
            training_cfg=training_cfg,
            sample_rate=int(cfg["data"]["sample_rate"]),
            discriminator_cfg=disc_cfg,
            schedule_cfg=sched_cfg,
            eval_metrics=_eval_metrics,
            test_save_dir=save_dir,
            test_max_audio=args.max_audio,
        )
    else:
        pl_module = SeparationModule.load_from_checkpoint(
            args.checkpoint, model=model,
            training_cfg=cfg["training"],
            sample_rate=int(cfg["data"]["sample_rate"]),
            eval_metrics=_eval_metrics,
            test_save_dir=save_dir,
            test_max_audio=args.max_audio,
        )

    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    results = trainer.test(pl_module, dataloaders=datamodule.test_dataloader()
                           if args.split == "test" else datamodule.val_dataloader(),
                           verbose=True)

    logger.info(f"results: {json.dumps(results, indent=2)}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
