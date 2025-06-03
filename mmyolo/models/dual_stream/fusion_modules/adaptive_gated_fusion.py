import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule # 假设您在使用 MMEngine 或类似框架

from mmyolo.registry import MODELS

@MODELS.register_module() # 如果您使用MMEngine/MMDetection的注册器，请取消注释
class AdaptiveGatedFusion(BaseModule):
    """
    Adaptive Gated Fusion Module for fusing two feature maps.

    Args:
        in_channels (int): Number of channels in the input feature maps.
        gate_intermediate_channels (int, optional): Number of intermediate channels in the
            gate generator network. If None, it's set to in_channels // reduction_ratio.
            Defaults to None.
        reduction_ratio (int): Reduction ratio for calculating gate_intermediate_channels.
            Defaults to 4.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """
    def __init__(
        self,
        in_channels: int,
        gate_intermediate_channels: int = None,
        reduction_ratio: int = 4,
        init_cfg = None # 用于 MMEngine BaseModule
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels

        if gate_intermediate_channels is None:
            # 确保门控网络的中间通道数不至于过小
            gate_intermediate_channels = max(16, in_channels // reduction_ratio)

        # 门控权重生成网络
        # 输入为拼接后的特征 (2 * in_channels)，输出为2个通道的门控图
        self.gate_generator = nn.Sequential(
            nn.Conv2d(in_channels * 2, gate_intermediate_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(gate_intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_intermediate_channels, 2, kernel_size=1, bias=True) # 输出2个通道，分别对应input1和input2的门
        )

        # 最终的卷积层，用于提炼融合后的特征
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        """
        Forward pass for Adaptive Gated Fusion.

        Args:
            input1 (Tensor): Feature tensor from modality 1. Shape: (B, C, H, W).
            input2 (Tensor): Feature tensor from modality 2. Shape: (B, C, H, W).
                             Assumes input1 and input2 have the same shape and number of channels.

        Returns:
            Tensor: Fused feature tensor. Shape: (B, C, H, W).
        """
        if input1.shape != input2.shape:
            raise ValueError(f"Input tensors must have the same shape, but got {input1.shape} and {input2.shape}")
        if input1.size(1) != self.in_channels:
            raise ValueError(f"Channel dimension of input1 ({input1.size(1)}) does not match in_channels ({self.in_channels})")
        if input2.size(1) != self.in_channels:
            raise ValueError(f"Channel dimension of input2 ({input2.size(1)}) does not match in_channels ({self.in_channels})")

        # 1. 拼接输入特征 (沿通道维度)
        combined_inputs = torch.cat([input1, input2], dim=1) # Shape: (B, 2*C, H, W)

        # 2. 生成门控权重
        # gates 的形状为 (B, 2, H, W)，这两个通道分别对应 input1 和 input2 的权重图
        gates = self.gate_generator(combined_inputs)

        # 3. 应用 Softmax 使每个空间位置的门控权重和为1 (形成加权平均)
        # g 的形状为 (B, 2, H, W)
        g = torch.softmax(gates, dim=1)
        
        gate1 = g[:, 0:1, :, :] # Shape: (B, 1, H, W)
        gate2 = g[:, 1:2, :, :] # Shape: (B, 1, H, W)

        # 4. 加权融合
        # 利用广播机制将 gate1 和 gate2 分别与 input1 和 input2 相乘
        fused_features_weighted = input1 * gate1 + input2 * gate2

        # 5. 最终提炼
        # 通过一个1x1卷积进一步融合和提炼特征
        refined_fused_features = self.relu(self.final_conv(fused_features_weighted))
        
        return refined_fused_features