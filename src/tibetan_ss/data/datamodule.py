"""Lightning DataModule that wires the dataset + dataloaders together."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as pl
import numpy as np
from torch.utils.data import DataLoader

from .dataset import TibetanMixDataset, collate_variable_length
from .mixing import MixingConfig


class TibetanMixDataModule(pl.LightningDataModule):
    """Lightning wrapper around :class:`TibetanMixDataset`.

    Expects ``cfg`` to be the resolved data-config (see
    ``configs/data/default.yaml``), and ``training_cfg`` for the DataLoader
    parameters (batch size, num workers).
    """

    def __init__(self, cfg: dict, training_cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.training_cfg = training_cfg
        self._train: TibetanMixDataset | None = None
        self._val: TibetanMixDataset | None = None
        self._test: TibetanMixDataset | None = None

    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None) -> None:
        out_root = Path(self.cfg["paths"]["output_root"])
        manifests = out_root / self.cfg["offline"]["manifest_subdir"]
        dm_on = bool(self.cfg.get("dynamic_mixing", {}).get("enabled", False))
        sr = int(self.cfg["sample_rate"])

        if stage in (None, "fit"):
            self._train = self._build_split("train", manifests, dm_on, sr)
            self._val   = self._build_split("val",   manifests, False, sr)
        if stage in (None, "test", "validate"):
            self._test  = self._build_split("test",  manifests, False, sr,
                                            full_length=True)
            if self._val is None:
                self._val = self._build_split("val", manifests, False, sr)

    # ------------------------------------------------------------------
    def _build_split(
        self,
        split: str,
        manifests_dir: Path,
        dynamic: bool,
        sr: int,
        full_length: bool = False,
    ) -> TibetanMixDataset:
        mcfg = self._mixing_cfg(split, sr, full_length=full_length)
        if dynamic and split == "train":
            return TibetanMixDataset(
                split=split,
                speakers=_load_json(manifests_dir / f"speakers_{split}.json"),
                noise_files=_load_json(manifests_dir / f"noise_{split}.json"),
                mixing_cfg=mcfg,
                dynamic=True,
                samples_per_epoch=int(self.cfg["dynamic_mixing"].get("cache_per_epoch", 20000)),
                seed=int(self.cfg["offline"]["seed"]) + _SPLIT_SEED[split],
            )
        return TibetanMixDataset(
            split=split,
            manifest_path=manifests_dir / f"{split}.json",
            seed=int(self.cfg["offline"]["seed"]) + _SPLIT_SEED[split],
        )

    # ------------------------------------------------------------------
    def _mixing_cfg(self, split: str, sr: int, full_length: bool = False) -> MixingConfig:
        seg = self.cfg["segment"]
        mix = self.cfg["mixing"]
        return MixingConfig(
            sample_rate=sr,
            segment_seconds=float(seg[split]) if seg[split] is not None else 0.0,
            random_length=bool(seg.get("random_length", True)) and not full_length,
            min_seconds=float(seg["min_seconds"]),
            max_seconds=float(seg["max_seconds"]),
            overlap=mix["overlap"][split],
            level_diff_db=mix["level_diff_db"],
            snr_db=mix["snr_db"][split],
            gender_pairing=str(mix.get("gender_pairing", "random")),
            rms_target_dbfs=float(mix.get("rms_target_dbfs", -25.0)),
            noise_enabled=bool(self.cfg["noise"]["enabled"]),
            noise_prob=float(self.cfg["noise"].get("prob_apply", 1.0)),
            full_length=full_length,
        )

    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=int(self.training_cfg["dataloader"]["batch_size"]),
            num_workers=int(self.training_cfg["dataloader"]["num_workers"]),
            pin_memory=bool(self.training_cfg["dataloader"]["pin_memory"]),
            persistent_workers=bool(self.training_cfg["dataloader"]["persistent_workers"]),
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=int(self.training_cfg["dataloader"]["batch_size"]),
            num_workers=int(self.training_cfg["dataloader"]["num_workers"]),
            pin_memory=bool(self.training_cfg["dataloader"]["pin_memory"]),
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test,
            batch_size=1,
            num_workers=int(self.training_cfg["dataloader"]["num_workers"]),
            pin_memory=False,
            shuffle=False,
            collate_fn=collate_variable_length,
        )


_SPLIT_SEED = {"train": 1, "val": 2, "test": 3}


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
