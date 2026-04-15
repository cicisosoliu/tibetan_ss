from .mixing import MixingConfig, MixtureSimulator, sample_overlap, sample_level_diff
from .dataset import TibetanMixDataset


def __getattr__(name):
    # Defer the heavy Lightning import until someone actually asks for the DataModule.
    if name == "TibetanMixDataModule":
        from .datamodule import TibetanMixDataModule as _cls
        return _cls
    raise AttributeError(name)
