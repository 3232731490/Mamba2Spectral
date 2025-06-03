import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule

from mmyolo.registry import MODELS

@MODELS.register_module()
class BackboneFusionModule(BaseModule):
    """主干网络融合模块：融合红外和可见光特征，并利用通道+空间注意力优化融合结果。"""
    def __init__(self, in_channels, reduction=16, init_cfg=None):
        super(BackboneFusionModule, self).__init__(init_cfg)
        self.in_channels = in_channels
        # 1x1 卷积融合两路特征
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        # 通道注意力（SE 机制）：自适应地重新标定通道响应:contentReference[oaicite:9]{index=9}
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # 空间注意力（CBAM 机制）：对融合特征的平均和最大池化结果做卷积映射生成空间权重:contentReference[oaicite:10]{index=10}
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_ir, feat_vis):
        # 拼接红外和可见光特征 (B, 2C, H, W)
        fused = torch.cat([feat_ir, feat_vis], dim=1)
        # 1x1 卷积降维融合 (B, C, H, W)
        fused = self.conv1(fused)
        fused = self.relu(fused)
        # 通道注意力机制
        attn_c = self.ca(fused)          # (B, C, 1, 1)
        fused = fused * attn_c          # 逐通道加权
        # 空间注意力机制：分别对加权后的特征做平均池化和最大池化
        avg = torch.mean(fused, dim=1, keepdim=True)  # (B, 1, H, W)
        max_ = torch.max(fused, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        attn_s = self.sa(torch.cat([avg, max_], dim=1))  # (B, 1, H, W)
        fused = fused * attn_s        # 逐位置加权
        return fused

@MODELS.register_module()
class DetectionHeadFusionModule(BaseModule):
    """检测头融合模块：融合红外和可见光特征，采用跨模态加权+SE注意力重校准融合特征。"""
    def __init__(self, in_channels, reduction=16, init_cfg=None):
        super(DetectionHeadFusionModule, self).__init__(init_cfg)
        self.in_channels = in_channels
        # 1x1 卷积：用于生成跨模态的通道权重
        self.conv_ir = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv_vis = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        # 通道注意力（SE 机制）：对融合特征进行重校准:contentReference[oaicite:11]{index=11}
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_ir, feat_vis):
        # 全局池化获取通道描述
        ir_pool = F.adaptive_avg_pool2d(feat_ir, 1)   # (B, C, 1, 1)
        vis_pool = F.adaptive_avg_pool2d(feat_vis, 1)
        # 跨模态通道加权：用可见光信息对红外特征加权，反之亦然
        w_ir = torch.sigmoid(self.conv_ir(vis_pool))  # (B, C, 1, 1)
        w_vis = torch.sigmoid(self.conv_vis(ir_pool))
        feat_ir_mod = feat_ir * w_ir
        feat_vis_mod = feat_vis * w_vis
        # 相加融合
        fused = feat_ir_mod + feat_vis_mod          # (B, C, H, W)
        fused = self.relu(fused)
        # 通道注意力重校准融合结果
        scale = self.se(fused)                     # (B, C, 1, 1)
        fused = fused * scale
        return fused
