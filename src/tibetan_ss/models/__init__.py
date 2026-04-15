"""Model zoo: unified interface for the baselines + our proposed model."""

from .base import BaseSeparator
from . import registry  # side-effect: registers every built-in model

build_model = registry.build_model
list_models = registry.list_models
