import torch
import torch.nn as nn
import torch.nn.functional as F
from mmyolo.registry import MODELS
class FeatureSplitting(nn.Module):
    def __init__(self, channels):
        super(FeatureSplitting, self).__init__()
        # 论文中提到 alpha 和 beta 是自适应参数，这里初始化为可训练的nn.Parameter
        # 实际训练中它们会通过反向传播进行更新
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))
        self.beta1 = nn.Parameter(torch.ones(1))
        self.beta2 = nn.Parameter(torch.ones(1))

    def forward(self, F_rgb, F_ir):
        # F_rgb 和 F_ir 的形状通常是 (B, C, H, W)
        B, C, H, W = F_rgb.shape

        # 1. 计算每个通道的余弦相似度
        # 将每个通道视为一个向量，进行展平操作 (C, H*W)
        F_rgb_flatten = F_rgb.view(B, C, -1) # (B, C, H*W)
        F_ir_flatten = F_ir.view(B, C, -1)   # (B, C, H*W)

        # 计算余弦相似度
        # 对每个通道向量计算范数 ||F_rgb|| 和 ||F_ir||
        norm_rgb = F_rgb_flatten.norm(p=2, dim=2, keepdim=True) # (B, C, 1)
        norm_ir = F_ir_flatten.norm(p=2, dim=2, keepdim=True)   # (B, C, 1)

        # 计算点积 F_rgb . F_ir
        dot_product = torch.sum(F_rgb_flatten * F_ir_flatten, dim=2, keepdim=True) # (B, C, 1)

        # 避免除以零
        denominator = (norm_rgb * norm_ir) + 1e-6
        
        # cos_m(theta) (B, C, 1)
        cos_theta_m = dot_product / denominator 
        
        # 将 cos_theta_m 扩展回 (B, C, 1, 1) 或 (B, C, H, W) 以便后续按通道相乘
        cos_theta_m_reshaped = cos_theta_m.view(B, C, 1, 1)

        # 2. 根据 cos_theta_m 构造 C_com 和 C_dif
        # 论文描述：C_com (值 > 0 设为 1，< 0 设为 0)
        C_com = (cos_theta_m_reshaped > 0).float()
        # 论文描述：C_dif (值 < 0 设为 1，> 0 设为 0)
        C_dif = (cos_theta_m_reshaped < 0).float()

        # 3. 应用自适应参数和矩阵进行特征加权
        F_com1 = self.alpha1 * C_com * F_rgb # (B, C, H, W) [cite: 132]
        F_com2 = self.alpha2 * C_com * F_ir  # (B, C, H, W) [cite: 132]
        F_dif1 = self.beta1 * C_dif * F_rgb  # (B, C, H, W) [cite: 132]
        F_dif2 = self.beta2 * C_dif * F_ir   # (B, C, H, W) [cite: 132]

        return F_com1, F_com2, F_dif1, F_dif2

class DistinctFeatureProcessing(nn.Module):
    def __init__(self, channels, groups=1): # 论文提到特征分组，这里设置为可配置参数
        super(DistinctFeatureProcessing, self).__init__()
        self.channels = channels
        self.groups = groups
        
        # 3x3 卷积分支
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # X 和 Y 方向的平均池化分支 (用于通道信息交互)
        # 这里池化后会展平，然后通过一个 Conv1 (可能是 1x1 卷积或者全连接层)
        # 论文图示中 Conv1 之后接 Sigmoid，再连接到 Re-weight
        # Re-weight 通常是与原始特征相乘
        
        # 假设图中的 Conv1 是一个 1x1 卷积，用于处理池化后的特征
        # X Avg Pool + Y Avg Pool -> Concat -> Conv1 -> Sigmoid
        
        # 降低维度以减少参数，然后恢复
        reduction_ratio = 16 # 经验值，论文未明确给出 Conv1 后的维度，但提到减少参数
        self.conv1_reducer = nn.Conv2d(channels * 2, channels // reduction_ratio, kernel_size=1)
        self.conv1_recover = nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1)
        
        # Cross-spatial learning 部分
        # AvgPool + Softmax, 然后进行交叉相乘
        # 这里的具体实现可以有多种解释，我将按一个常见的空间注意力机制来模拟
        # 假设通过 AvgPool 得到空间注意力图
        self.spatial_attn_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # DFP 模块的最终输出是通过 Re-weight 得到的，通常 Re-weight 是将注意力图与原始特征相乘
        # 在这里我们先将 dif1 和 dif2 相加，然后进行处理，最后再与原始特征进行加权
        # 论文图示是 (F_dif1 + F_dif2) 经过 DFP 后得到 F_dif，F_dif 再和 F_com 融合
        # 这意味着 DFP 内部的 Attention 是基于 F_dif1 和 F_dif2 聚合的信息
        
        # 重新权重 (Re-weight) 操作的 Sigmoid 激活函数
        self.reweight_sigmoid = nn.Sigmoid()

    def forward(self, F_dif1, F_dif2):
        B,C,H,W = F_dif1.shape
        # 初始融合
        F_dif_sum = F_dif1 + F_dif2 # (B, C, H, W)
        
        # 3x3 卷积分支
        conv_out = self.conv3x3(F_dif_sum) # (B, C, H, W)
        
        # X & Y Avg Pool + Concat + Conv1 + Sigmoid 分支 (通道注意力)
        # 沿 H 维度平均池化 (B, C, 1, W)
        x_avg_pool = F.adaptive_avg_pool2d(F_dif_sum, (1, F_dif_sum.shape[3]))
        # 沿 W 维度平均池化 (B, C, H, 1)
        y_avg_pool = F.adaptive_avg_pool2d(F_dif_sum, (F_dif_sum.shape[2], 1))

        # 展平并拼接
        # 将 x_avg_pool 展平为 (B, C, W) 并转置为 (B, C, W) (为了和 y_avg_pool 维度一致，因为 y_avg_pool是 (B, C, H))
        # 实际上，这里需要仔细看图。图示是 (B, C/G, 1, W) 和 (B, C/G, H, 1) 拼接为 (B, C/G, H+W)
        # 这里需要将 x_avg_pool 展平为 (B, C, W)
        # 将 y_avg_pool 展平为 (B, C, H)
        # 然后将它们沿最后一个维度拼接，变为 (B, C, H+W)
        
        x_avg_pool_flat = x_avg_pool.squeeze(2) # (B, C, W)
        y_avg_pool_flat = y_avg_pool.squeeze(3) # (B, C, H)
        
        # 为了拼接，需要将 H 和 W 维度统一到一个维度，然后通过 1x1 卷积处理
        # 这里的图示更像是一个 ECA-Net (Efficient Channel Attention) 变体或者 Coordinate Attention
        # 论文图示 Concat (C/G * 1 * W + C/G * H * 1) 意味着对 X 和 Y 的平均池化结果进行拼接
        # 然后通过一个 Conv1 (1x1卷积) 处理，再接 Sigmoid
        
        # 重新调整形状以进行拼接，并适应 Conv2d
        x_avg_pool_reshaped = x_avg_pool.permute(0, 1, 3, 2) # (B, C, W, 1) for Conv2d
        y_avg_pool_reshaped = y_avg_pool # (B, C, H, 1) for Conv2d

        # 为了拼接，需要维度一致
        # 我们假设论文中的 Conv1 是处理通道维度的信息，它将 (B, C, H, W) 的全局信息聚合
        # 最直接的理解：将 x_avg_pool 和 y_avg_pool 在通道维度上进行拼接
        # 或者更接近图示：x_avg_pool 和 y_avg_pool 经过处理后，再 concat
        
        # 根据图示，Concat 后会接 Conv1
        # 假设 X Avg Pool 和 Y Avg Pool 后的输出被 Concat
        # 然后 Conv1 应该是一个 1x1 卷积，用于降维和信息聚合，再接 Sigmoid
        
        # 简化处理：将 x_avg_pool 和 y_avg_pool 的特征拼接后，通过 Conv1 处理
        # 这更像是 Coordinate Attention 的简化版
        x_channel_attn = F.adaptive_avg_pool2d(F_dif_sum, (H, 1)) # (B, C, H, 1)
        y_channel_attn = F.adaptive_avg_pool2d(F_dif_sum, (1, W)) # (B, C, 1, W)
        
        # 将两个特征图拼接后通过一个共享的 1x1 卷积
        # 论文图示中的 Concat (C/G * 1 * W + C/G * H * 1) 可能是指将两个池化结果展平后拼接
        # 让我们按照 Coordinate Attention 的思路，但简化实现
        
        # 空间注意力 (Cross-spatial learning)
        # 论文图示是 AvgPool + Softmax 进行交叉相乘
        # F_dif_sum -> AvgPool (H, W) -> Softmax
        # 得到空间注意力图
        spatial_attention_map = self.spatial_attn_conv(F_dif_sum)
        spatial_attention_map = self.sigmoid(spatial_attention_map) # (B, 1, H, W)

        # Re-weighting
        # 论文图示中 Re-weight 是将注意力矩阵和原始特征相乘
        # 在这里，我们将所有分支的信息融合，然后得到一个最终的注意力图来加权 F_dif_sum
        
        # DFP 模块的输出 F_dif = (conv_out + 空间注意力加权后的 F_dif_sum) * 一个最终的 sigmoid 激活
        # 这里对图示进行了一定程度的解释和简化，因为细节未完全提供
        
        # 最终的 F_dif 通常是原始特征经过注意力加权后的结果
        # 这里，我们将 Conv 3x3 的输出与一个通过 Sigmoid 得到的注意力图相乘
        
        # 将 conv_out 视为主要的增强路径
        # 空间注意力图应用于 conv_out
        
        F_dif = conv_out * spatial_attention_map # (B, C, H, W)
        
        # 论文图示在 DFP 的最后还有一个 Sigmoid 和 Re-weight，这通常意味着一个门控机制或最终的归一化
        # 在这里，我们假设 DFP 的输出是融合了所有分支信息并增强后的特征
        # 如果需要严格按照图示的最后一个 Re-weight，可以这样：
        # final_attention = self.reweight_sigmoid(F_dif)
        # F_dif_output = F_dif_sum * final_attention # 乘以原始融合的输入
        
        # 考虑到 DFP 模块的目标是增强特定特征，这里的 F_dif 可以直接是增强后的特征
        return F_dif

class SimilarFeatureProcessing(nn.Module):
    def __init__(self, channels):
        super(SimilarFeatureProcessing, self).__init__()
        
        self.channels = channels
        
        # 共享的全连接层 f_FC
        # 论文提到共享第一层权重，并降维到 1/32
        # 这里我们实现为一个序列模块，模拟共享 FC 层
        reduction_ratio = 32
        self.shared_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False), # 降维
            nn.ReLU(inplace=True),
        )
        
        # 独立的 FC 层用于 z1 和 z2 (论文中 f_FC1 和 f_FC2 共享第一层)
        # 这里，我们用两个独立的线性层来表示 z1 和 z2 的生成
        self.fc_z1 = nn.Linear(channels // reduction_ratio, channels, bias=False)
        self.fc_z2 = nn.Linear(channels // reduction_ratio, channels, bias=False)

    def forward(self, F_com1, F_com2):
        # 1. 元素级求和 F = F_com1 + F_com2
        F_sum = F_com1 + F_com2 # (B, C, H, W) [cite: 147]

        # 2. 全局平均池化 s = f_GAP(F)
        s = F.adaptive_avg_pool2d(F_sum, (1, 1)).squeeze(-1).squeeze(-1) # (B, C) [cite: 147]

        # 3. 通过共享的 FC 层生成 z1 和 z2
        shared_out = self.shared_fc(s) # (B, C // reduction_ratio)
        
        z1 = self.fc_z1(shared_out) # (B, C) [cite: 147]
        z2 = self.fc_z2(shared_out) # (B, C) [cite: 147]

        # 4. Softmax 操作得到注意力向量 a 和 b
        # 论文图示是独立 Softmax, 但考虑到两个 attention vectors 通常是互补的，
        # 常见 SKNet 中对两个分支进行 concat 然后一个 softmax，或者对每个分支独立 softmax
        # 论文明确说是 "individual softmax operations for each modality" [cite: 143]
        # 但是，为了确保 a+b=1 的选择特性（如 SKNet），通常会在 z1 和 z2 拼接后进行 softmax
        # 论文公式 (11) 和 (12) 独立应用 f_softmax，这意味着 a 和 b 分别归一化。
        # 如果是这样，它们不一定互补。我们遵循论文的公式描述。
        
        # 为了与原始特征进行乘法，需要将 attention 向量扩展为 (B, C, 1, 1)
        a = F.softmax(z1, dim=1).view(z1.shape[0], self.channels, 1, 1) # (B, C, 1, 1) [cite: 147]
        b = F.softmax(z2, dim=1).view(z2.shape[0], self.channels, 1, 1) # (B, C, 1, 1) [cite: 147]

        # 5. 元素级相乘和求和得到 F_com
        F_com = a * F_com1 + b * F_com2 # (B, C, H, W) [cite: 147]

        return F_com
from mmengine.model import BaseModule
@MODELS.register_module()
class CSIFF_ge(BaseModule):
    def __init__(self, in_channels):
        super(CSIFF_ge, self).__init__()
        self.fs_module = FeatureSplitting(in_channels)
        self.sfp_module = SimilarFeatureProcessing(in_channels)
        self.dfp_module = DistinctFeatureProcessing(in_channels) # 假设 DFP 内部处理了分组

    def forward(self, F_rgb, F_ir):
        # 1. Feature Splitting (FS)
        F_com1, F_com2, F_dif1, F_dif2 = self.fs_module(F_rgb, F_ir)

        # 2. Similar Feature Processing (SFP)
        F_com = self.sfp_module(F_com1, F_com2)

        # 3. Distinct Feature Processing (DFP)
        F_dif = self.dfp_module(F_dif1, F_dif2)

        # 4. Combine processed features (element-wise addition)
        F_fused = F_com + F_dif # 论文图示是元素级相加 [cite: 118, 119]

        return F_fused