import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mamba import MM_SS2D
from dmff_fusion_my import TransformerFusionBlock

@MODELS.register_module()
class MambaTransformerFusion(BaseModule):
    def __init__(self,
                 in_channels: int,
                 size: tuple,
                 mamba_operations: int = 1,
                 vert_anchor: int = 16,
                 horz_anchor: int = 16,
                 loops_num: int = 1,
                 fusion: bool = True):
        """
        Args:
            in_channels (int): 输入图像的通道数（C）。
            size (tuple): 输入图像尺寸 (H, W)，用于 MM_SS2D 的 PatchEmbed。
            mamba_operations (int): MM_SS2D 中循环执行 SS2D_intra/inter 的次数。
            vert_anchor (int): TransformerFusionBlock 的垂直采样尺度。
            horz_anchor (int): TransformerFusionBlock 的水平采样尺度。
            loops_num (int): TransformerFusionBlock 中交叉注意力循环次数。
            fusion (bool): 是否输出融合后的单一图像（True）或两路增强图像（False）。
        """
        super().__init__()
        # 空间结构建模模块：输入两模态图像，输出两路增强特征
        self.mamba = MM_SS2D(in_channels=in_channels, size=size, mamba_opreations=mamba_operations)
        # 跨模态融合模块：对两路特征进行交叉注意力融合
        self.transformer = TransformerFusionBlock(
            d_model=in_channels,
            vert_anchors=vert_anchor,
            horz_anchors=horz_anchor,
            fusion=fusion,
            loops_num=loops_num
        )

    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor):
        """
        Args:
            inputs1, inputs2 (torch.Tensor): 两路输入图像，shape 为 (B, C, H, W)。
        Returns:
            list: 如果 fusion=True，则返回融合后的单个图像列表 [fused_feature]；
                  如果 fusion=False，则返回两路增强图像 [enhanced1, enhanced2]。
        """
        # 1. 空间结构建模：得到两路增强特征
        feat1, feat2 = self.mamba(inputs1, inputs2)
        # 2. 跨模态融合：交叉注意力融合特征
        out = self.transformer(feat1, feat2)
        return out

if __name__ == "__main__":
    # 测试代码
    model = MambaTransformerFusion(in_channels=3, size=(224, 224), mamba_operations=1, vert_anchor=16, horz_anchor=16, loops_num=1, fusion=True)
    inputs1 = torch.randn(1, 3, 224, 224)
    inputs2 = torch.randn(1, 3, 224, 224)
    print(inputs1.shape, inputs2.shape)  # 输出输入图像形状
    output = model(inputs1, inputs2)
    print(output[0].shape)  # 输出融合后的特征图形状