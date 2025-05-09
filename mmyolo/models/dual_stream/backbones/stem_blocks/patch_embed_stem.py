
from mmcv.cnn import ConvModule

from mmyolo.registry import MODELS
from mmengine.model import BaseModule
from mmyolo.models.utils import make_divisible, make_round

import torch
import torch.nn as nn

from mmyolo.models.dual_stream.backbones.backbone_pool.mobileMamba import Conv2d_BN

@MODELS.register_module()
class PatchEmbedStem(BaseModule):
    """PatchEmbedStem模块,用于YOLOv8模型的stem部分。

    Args:
        widen_factor (float): 通道宽度倍率,默认为1.0。
        in_channels (int): 输入通道数,默认为3。
        out_channels (int): 输出通道数,默认为64。
    """

    def __init__(self,
                 widen_factor: float = 1.0,
                 in_channels: int = 3,
                 out_channels: int = 64):
        super().__init__()

        out_channels = make_divisible(out_channels, widen_factor)

        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_channels, out_channels // 2, 3, 2, 1),
                                               torch.nn.ReLU(),
                                               Conv2d_BN(out_channels // 2, out_channels, 3, 2, 1,
                                                         ))
        
    def forward(self, x):
        """前向传播函数。
        """
        return self.patch_embed(x)