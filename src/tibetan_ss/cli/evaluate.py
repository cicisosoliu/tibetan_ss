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
    ModuleCls = ProposedGANModule if engine_name == "gan" else SeparationModule
    pl_module = ModuleCls.load_from_checkpoint(
        args.checkpoint, model=model,
        training_cfg=cfg["training"],
        sample_rate=int(cfg["data"]["sample_rate"]),
    )

    trainer = pl.Trainer(accelerator="auto", devices="auto", logger=False)
    results = trainer.test(pl_module, dataloaders=datamodule.test_dataloader()
                           if args.split == "test" else datamodule.val_dataloader(),
                           verbose=True)

    logger.info(f"results: {json.dumps(results, indent=2)}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
