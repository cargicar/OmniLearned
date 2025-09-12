"Script taken from https://arxiv.org/pdf/2509.07700"

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torchinfo import summary

from src.utils import get_logger, import_class_by_name

logger = get_logger()

class DiffusionModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        self.config = None

    def forward(self, *args, **kwargs):
        """Forward pass computes the loss. Should be defined by the inherited class."""
        raise NotImplementedError

    def sample(self):
        """Inference pipeline. Should be defined by the inherited class."""
        raise NotImplementedError

    @property
    def device(self):
        return next(self.model.parameters()).device
    
    def save_config(self, config: dict) -> None:
        self.config = config

    def save_state(self, save_path: Union[str, Path]) -> None:
        torch.save({"model": self.model.state_dict(), "config": self.config}, save_path)
        logger.info(f"Saved model to {save_path}")

    def load_state(self, load_path: Union[str, Path]) -> None:
        state = torch.load(load_path, map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.config = state["config"]
        logger.info(f"Loaded model from {load_path}")

    def summarize(self):
        summary(self.model, input_data=self.model.example_input, col_names=["input_size", "output_size", "num_params"])



def create_model_from_config(config) -> DiffusionModelBase:
    architecture_cls = import_class_by_name(config["architecture_class"])
    assert issubclass(architecture_cls, nn.Module), f"Architecture class {architecture_cls} must be a subclass of nn.Module."
    
    diffusion_cls = import_class_by_name(config["diffusion_class"])
    assert issubclass(diffusion_cls, DiffusionModelBase), f"Diffusion class {diffusion_cls} must be a subclass of DiffusionModelBase."

    architecture_args = config.get("architecture_args", {})
    diffusion_args = config.get("diffusion_args", {})

    model_path = config.get("model_path")

    if model_path is not None:
        logger.info(f"Loading model from {model_path}")
        state = torch.load(model_path, map_location="cpu")

        saved_model_config = state["config"]["model"]
        saved_architecture_args = saved_model_config["architecture_args"]
        saved_diffusion_args = saved_model_config["diffusion_args"]

        # merge passed config with the saved config
        architecture_args = OmegaConf.merge(saved_architecture_args, architecture_args)
        diffusion_args = OmegaConf.merge(saved_diffusion_args, diffusion_args)

    architecture: nn.Module = architecture_cls(**architecture_args)
    model: DiffusionModelBase = diffusion_cls(model=architecture, **diffusion_args)

    if model_path is not None:
        model.load_state(model_path)

    model.eval()

    return model
