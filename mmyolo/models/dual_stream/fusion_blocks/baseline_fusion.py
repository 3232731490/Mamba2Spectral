import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import numpy as np

from typing import Tuple , List
from torch import Tensor

from mmengine.model import BaseModule
from mmyolo.models.utils import make_divisible, make_round

from mmyolo.registry import MODELS

@MODELS.register_module()
class BaseFusion(BaseModule):
    def __init__(
        self,
        in_channels: int,
    ) -> None:
        super().__init__()
        # self.block1 = nn.Conv2d(in_channels * 2 , in_channels , 1, 1,0)
        # self.block2 = nn.Conv2d(in_channels * 2 , in_channels , 1, 1,0)
        # self.fusion_blocks = nn.Sequential(*blocks)
        
    def forward(self, input1: Tuple[Tensor],input2: Tuple[Tensor]):
        # ir_fea = self.block1(torch.cat([input1,input2] , dim = 1))
        # vis_fea = self.block2(torch.cat([input2,input1] , dim = 1))
        """Forward function."""
        return input1 + input2

    
@MODELS.register_module()
class BaseFusion2(BaseModule):
    def __init__(
        self,
        in_channels: int,
    ) -> None:
        super().__init__()
        # self.block1 = nn.Conv2d(in_channels * 2 , in_channels , 1, 1,0)
        # self.block2 = nn.Conv2d(in_channels * 2 , in_channels , 1, 1,0)
        # self.fusion_blocks = nn.Sequential(*blocks)
        
    def forward(self, input1: Tuple[Tensor],input2: Tuple[Tensor]):
        # ir_fea = self.block1(torch.cat([input1,input2] , dim = 1))
        # vis_fea = self.block2(torch.cat([input2,input1] , dim = 1))
        """Forward function."""
        return [input1 + input2]

