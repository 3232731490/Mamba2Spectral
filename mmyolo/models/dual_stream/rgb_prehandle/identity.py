import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from mmengine.model import BaseModule
from mmyolo.registry import MODELS

@MODELS.register_module()
class Identity(BaseModule):
    """Identity module for RGB pre-processing."""

    def __init__(self) -> None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, inputs: torch.Tensor):
        """Forward function."""
        return self.identity(inputs)