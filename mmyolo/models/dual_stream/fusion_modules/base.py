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
class Add(nn.Module):
    #  Add two tensors
    def __init__(self,):
        super(Add, self).__init__()


    def forward(self, x1: torch.Tensor,x2:torch.Tensor):
        return torch.add(x1, x2)
    
@MODELS.register_module()
class CommonConv(nn.Module):
    def __init__(self,in_channels):
        super(CommonConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels * 2,
                      out_channels=in_channels,
                      kernel_size = 3,
                      padding='same',
                      bias=False)

    def forward(self, x1: torch.Tensor,x2:torch.Tensor):
        return self.conv(torch.cat([x1,x2],dim=1))