import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F
from .common import SimConv,AdvPoolFusion,SimFusion_3in,SimFusion_4in,SimFusion_2in
import numpy as np
from .function import conv_bn,get_avg_pool,onnx_AdaptiveAvgPool2d,get_shape

import math
from typing import List, Union

from mmyolo.registry import MODELS

from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.dual_stream.fusion_modules.Wconv.wconv import WTConv

__all__ = ('GoldYoloNeck')

class MLPBlock(nn.Module):
    """Implements a single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function."""
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLPBlock."""
        return self.lin2(self.act(self.lin1(x)))

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        
        self.add_module('c', nn.Conv2d(
                a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert kernel_size == 3
        assert padding == 1
        
        padding_11 = padding - kernel_size // 2
        
        self.nonlinearity = nn.ReLU()
        
        self.se = nn.Identity()

        self.rbr_identity = nn.BatchNorm2d(
                    num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
        self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
    
    def forward(self, inputs):
        '''Forward process'''
        id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
    
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    
    def forward(self, x):
        return self.relu(x + 3) / 6


class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            global_inp=None,
    ) -> None:
        super().__init__()
        
        if not global_inp:
            global_inp = inp
        ### TODO 调整参数
        self.local_embedding = Conv(inp, oup, k=1,s=1,p=0,g=1,d=1)
        self.global_embedding = Conv(global_inp, oup, k=1,s=1,p=0,g=1,d=1)
        self.global_act = Conv(global_inp, oup, k=1,s=1,p=0,g=1,d=1)
        self.act = h_sigmoid()
    
    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H
        
        local_feat = self.local_embedding(x_l)
        
        global_act = self.global_act(x_g)
        global_feat = self.global_embedding(x_g)
        
        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])
            
            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)
        
        else:
            sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
            global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        out = local_feat * sig_act + global_feat
        return out

class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()
        
        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class PyramidPoolAgg(nn.Module):
    def __init__(self, stride, pool_mode='onnx'):
        super().__init__()
        self.stride = stride
        if pool_mode == 'torch':
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == 'onnx':
            self.pool = onnx_AdaptiveAvgPool2d
    
    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        
        output_size = np.array([H, W])
        
        if not hasattr(self, 'pool'):
            self.pool = nn.functional.adaptive_avg_pool2d
        
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        
        out = [self.pool(inp, output_size) for inp in inputs]
        
        return torch.cat(out, dim=1)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class GD_DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(GD_DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GD_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        
        self.to_q = Conv2d_BN(dim, nh_kd, 1)
        self.to_k = Conv2d_BN(dim, nh_kd, 1)
        self.to_v = Conv2d_BN(dim, self.dh, 1)
        
        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
                self.dh, dim, bn_weight_init=0))
    
    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)
        
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)
        
        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k
        
        xx = torch.matmul(attn, vv)
        
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx

class top_Block(nn.Module):
    
    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.attn = GD_Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer,)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = GD_DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,)
    
    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class TopBasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.ReLU6):
        super().__init__()
        self.block_num = block_num
        
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(top_Block(
                    embedding_dim, key_dim=key_dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                    drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,))
    
    def forward(self, x):
        # token * N 
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x

class RepGDNeck_3(nn.Module):
    def __init__(
            self,
            channels_list=[1024,512,256,128],
            num_repeats=[12, 12, 12, 12],
            ### TODO 修改参数
            depths=2,
            fusion_in=1920,
            fuse_block_num=3,
            embed_dim_p=256,
            embed_dim_n=1408,
            trans_channels=[256, 128, 256, 512],
            key_dim=8,
            num_heads=4,
            mlp_ratios=1,
            attn_ratios=2,
            c2t_stride=2,
            drop_path_rate=0.1,
            pool_mode='torch',
            block=RepVGGBlock,
    ):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        
        self.low_FAM = SimFusion_4in()  # 特征对齐
        self.low_FAM_2 = SimFusion_4in()  # 特征对齐


        # 特征融合      最后输出通道数为：p3+p4；方便后续分割
        self.low_IFM = nn.Sequential(
                Conv(fusion_in, embed_dim_p, k=1, s=1, p=0),
                *[block(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
                Conv(embed_dim_p, sum(trans_channels[0:2]), k=1, s=1, p=0),
        )
        self.low_IFM_2 = nn.Sequential(
                Conv(fusion_in, embed_dim_p, k=1, s=1, p=0),
                *[block(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
                Conv(embed_dim_p, sum(trans_channels[0:2]), k=1, s=1, p=0),
        )

        # 改变C5的通道数 -- 1024 -> 512
        self.reduce_layer_c5 = SimConv(
                in_channels=channels_list[0],  # 1024
                out_channels=channels_list[1],  # 512
                kernel_size=1,
                stride=1
        )
        self.reduce_layer_c5_2 = SimConv(
                in_channels=channels_list[0],  # 1024
                out_channels=channels_list[1],  # 512
                kernel_size=1,
                stride=1
        )

        # 融合特征图 （C3、C4、C5） 作为F_local输入
        self.LAF_p4 = SimFusion_3in(
                in_channel_list=[channels_list[2], channels_list[1]],  # 512, 256       / 512 512
                out_channels=channels_list[1],  # 512
        )
        self.LAF_p4_2 = SimFusion_3in(
                in_channel_list=[channels_list[2], channels_list[1]],  # 512, 256       / 512 512
                out_channels=channels_list[1],  # 512
        )

        # 信息注入      将F_local 与 split后的第一个特征图融合
        self.Inject_p4 = InjectionMultiSum_Auto_pool(channels_list[1], channels_list[1],)
        self.Inject_p4_2 = InjectionMultiSum_Auto_pool(channels_list[1], channels_list[1],)
        
        # 进一步提取和融合信息
        self.Rep_p4 = RepBlock(
                in_channels=channels_list[1],  # 256
                out_channels=channels_list[1],  # 256
                n=num_repeats[0],
                block=block
        )
        self.Rep_p4_2 = RepBlock(
                in_channels=channels_list[1],  # 256
                out_channels=channels_list[1],  # 256
                n=num_repeats[0],
                block=block
        )


        # 改变P4的通道数： 256 -> 128
        self.reduce_layer_p4 = SimConv(
                in_channels=channels_list[1],  # 256
                out_channels=channels_list[2],  # 128
                kernel_size=1,
                stride=1
        )
        self.reduce_layer_p4_2 = SimConv(
                in_channels=channels_list[1],  # 256
                out_channels=channels_list[2],  # 128
                kernel_size=1,
                stride=1
        )
        
        # 融合特征图 ： （C2、C3、P4） 作为F_local输入
        self.LAF_p3 = SimFusion_3in(
                in_channel_list=[channels_list[3], channels_list[2]],  # 512, 256       / 256 256
                out_channels=channels_list[2],  # 256   / 128
        )
        self.LAF_p3_2 = SimFusion_3in(
                in_channel_list=[channels_list[3], channels_list[2]],  # 512, 256       / 256 256
                out_channels=channels_list[2],  # 256   / 128
        )

        # 信息注入      将F_local 与 split后的第二个特征图融合
        self.Inject_p3 = InjectionMultiSum_Auto_pool(channels_list[2], channels_list[2])
        self.Inject_p3_2 = InjectionMultiSum_Auto_pool(channels_list[2], channels_list[2])
        
        # 进一步提取和融合信息
        self.Rep_p3 = RepBlock(
                in_channels=channels_list[2],  # 128
                out_channels=channels_list[2],  # 128
                n=num_repeats[1],
                block=block
        )
        self.Rep_p3_2 = RepBlock(
                in_channels=channels_list[2],  # 128
                out_channels=channels_list[2],  # 128
                n=num_repeats[1],
                block=block
        )
        
        # 特征对齐 对所有输入特征图进行相同的池化操作来达到目的（这里很奇怪，图上并没有对p5进行池化？）
        self.high_FAM = PyramidPoolAgg(stride=c2t_stride, pool_mode=pool_mode)
        self.high_FAM_2 = PyramidPoolAgg(stride=c2t_stride, pool_mode=pool_mode)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        # 信息融合      一共有block_num个块堆叠而成 MHA+FFN
        self.high_IFM = TopBasicLayer(
                block_num=depths,
                embedding_dim=embed_dim_n,
                key_dim=key_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                attn_ratio=attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
        )
        self.high_IFM_2 = TopBasicLayer(
                block_num=depths,
                embedding_dim=embed_dim_n,
                key_dim=key_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                attn_ratio=attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
        )

        # 调整通道数，方便后续split
        self.conv_1x1_n = nn.Conv2d(embed_dim_n, sum(trans_channels[2:4]), 1, 1, 0)
        self.conv_1x1_n_2 = nn.Conv2d(embed_dim_n, sum(trans_channels[2:4]), 1, 1, 0)
        
        # 融合特征图（p3，p4）  用作F_local的输入
        self.LAF_n4 = AdvPoolFusion()
        self.LAF_n4_2 = AdvPoolFusion()

        # 信息注入 将F_local 与 split后的第一个特征图融合
        self.Inject_n4 = InjectionMultiSum_Auto_pool(channels_list[1], channels_list[1],)
        self.Inject_n4_2 = InjectionMultiSum_Auto_pool(channels_list[1], channels_list[1],)

        # 进一步提取和融合信息
        self.Rep_n4 = RepBlock(
                in_channels=channels_list[1],  # 128 + 128
                out_channels=channels_list[1],  # 256
                n=num_repeats[2],
                block=block
        )
        self.Rep_n4_2 = RepBlock(
                in_channels=channels_list[1],  # 128 + 128
                out_channels=channels_list[1],  # 256
                n=num_repeats[2],
                block=block
        )
        
        # 融合特征图（n4,c5） 用作F_local的输入
        self.LAF_n5 = AdvPoolFusion()
        self.LAF_n5_2 = AdvPoolFusion()

        # 信息注入      将F_local 与 split后的第二个特征图融合
        self.Inject_n5 = InjectionMultiSum_Auto_pool(channels_list[0], channels_list[0],)
        self.Inject_n5_2 = InjectionMultiSum_Auto_pool(channels_list[0], channels_list[0],)
        
        # 进一步提取和融合信息
        self.Rep_n5 = RepBlock(
                in_channels=channels_list[0],  # 256 + 256
                out_channels=channels_list[0],  # 512
                n=num_repeats[3],
                block=block
        )
        self.Rep_n5_2 = RepBlock(
                in_channels=channels_list[0],  # 256 + 256
                out_channels=channels_list[0],  # 512
                n=num_repeats[3],
                block=block
        )
        
        self.trans_channels = trans_channels
    
    def forward(self, input,input2):
        (c2, c3, c4, c5) = input
        (t2 , t3, t4, t5) = input2

        low_align_feat = self.low_FAM(input)
        low_align_feat2 = self.low_FAM_2(input2)

        low_fuse_feat = self.low_IFM(low_align_feat)
        low_fuse_feat2 = self.low_IFM_2(low_align_feat2)

        low_global_info = low_fuse_feat.split(self.trans_channels[0:2], dim=1)
        low_global_info2 = low_fuse_feat2.split(self.trans_channels[0:2], dim=1)
        
        ## inject low-level global info to p4
        c5_half = self.reduce_layer_c5(c5)
        c5_half2 = self.reduce_layer_c5_2(t5)

        p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
        p4_adjacent_info2 = self.LAF_p4_2([t3, t4, c5_half2])

        p4 = self.Inject_p4(p4_adjacent_info, low_global_info2[0])
        p42 = self.Inject_p4_2(p4_adjacent_info2, low_global_info[0])

        p4 = self.Rep_p4(p4)
        p42 = self.Rep_p4_2(p42)
        
        ## inject low-level global info to p3
        p4_half = self.reduce_layer_p4(p4)
        p4_half2 = self.reduce_layer_p4_2(p42)

        p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
        p3_adjacent_info2 = self.LAF_p3_2([t2, t3, p4_half2])

        p3 = self.Inject_p3(p3_adjacent_info, low_global_info2[1])
        p32 = self.Inject_p3_2(p3_adjacent_info2, low_global_info[1])

        p3 = self.Rep_p3(p3)
        p32 = self.Rep_p3_2(p32)
        
        # High-GD
        ## use transformer fusion global info
        high_align_feat = self.high_FAM([p3, p4, c5])
        high_align_feat2 = self.high_FAM_2([p32, p42, t5])

        high_fuse_feat = self.high_IFM(high_align_feat)
        high_fuse_feat2 = self.high_IFM_2(high_align_feat2)

        high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
        high_fuse_feat2 = self.conv_1x1_n_2(high_fuse_feat2)

        high_global_info = high_fuse_feat.split(self.trans_channels[2:4], dim=1)
        high_global_info2 = high_fuse_feat2.split(self.trans_channels[2:4], dim=1)

        ## inject low-level global info to n4
        n4_adjacent_info = self.LAF_n4(p3, p4_half)
        n4_adjacent_info2 = self.LAF_n4_2(p32, p4_half2)

        n4 = self.Inject_n4(n4_adjacent_info, high_global_info2[0])
        n42 = self.Inject_n4_2(n4_adjacent_info2, high_global_info[0])

        n4 = self.Rep_n4(n4)
        n42 = self.Rep_n4_2(n42)
        
        ## inject low-level global info to n5
        n5_adjacent_info = self.LAF_n5(n4, c5_half)
        n5_adjacent_info2 = self.LAF_n5_2(n42, c5_half2)

        n5 = self.Inject_n5(n5_adjacent_info, high_global_info2[1])
        n52 = self.Inject_n5_2(n5_adjacent_info2, high_global_info[1])
        n5 = self.Rep_n5(n5)
        n52 = self.Rep_n5_2(n52)
        
        outputs = [p3, n4, n5]
        outputs2 = [p32, n42, n52]
        return [ outputs, outputs2 ]
    
class RepGDNeck_4(nn.Module):
    def __init__(
            self,
            channels_list=[1024,512,256,128],
            num_repeats=[12, 12, 12, 12],
            ### TODO 修改参数
            depths=2,
            fusion_in=1920,
            fuse_block_num=3,
            embed_dim_p=256,
            embed_dim_n=1408,
            trans_channels=[256, 128, 64, 128,256, 512],
            key_dim=8,
            num_heads=4,
            mlp_ratios=1,
            attn_ratios=2,
            c2t_stride=2,
            drop_path_rate=0.1,
            pool_mode='torch',
            block=RepVGGBlock,
    ):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        
        self.low_FAM = SimFusion_4in()  # 特征对齐
        self.low_FAM_2 = SimFusion_4in()  # 特征对齐

        # 特征融合      最后输出通道数为：p3+p4；方便后续分割
        self.low_IFM = nn.Sequential(
                Conv(fusion_in, embed_dim_p, k=1, s=1, p=0),
                *[block(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
                Conv(embed_dim_p, sum(trans_channels[0:3]), k=1, s=1, p=0),
        )
        # 特征融合      最后输出通道数为：p3+p4；方便后续分割
        self.low_IFM_2 = nn.Sequential(
                Conv(fusion_in, embed_dim_p, k=1, s=1, p=0),
                *[block(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
                Conv(embed_dim_p, sum(trans_channels[0:3]), k=1, s=1, p=0),
        )
        # 改变C5的通道数 -- 1024 -> 512
        self.reduce_layer_c5 = SimConv(
                in_channels=channels_list[0],  # 1024
                out_channels=channels_list[1],  # 512
                kernel_size=1,
                stride=1
        )
        # 改变C5的通道数 -- 1024 -> 512
        self.reduce_layer_c5_2 = SimConv(
                in_channels=channels_list[0],  # 1024
                out_channels=channels_list[1],  # 512
                kernel_size=1,
                stride=1
        )

        # 融合特征图 （C3、C4、C5） 作为F_local输入
        self.LAF_p4 = SimFusion_3in(
                in_channel_list=[channels_list[2], channels_list[1]],  # 512, 256       / 512 512
                out_channels=channels_list[1],  # 512
        )
        # 融合特征图 （C3、C4、C5） 作为F_local输入
        self.LAF_p4_2 = SimFusion_3in(
                in_channel_list=[channels_list[2], channels_list[1]],  # 512, 256       / 512 512
                out_channels=channels_list[1],  # 512
        )

        # 信息注入      将F_local 与 split后的第一个特征图融合
        self.Inject_p4 = InjectionMultiSum_Auto_pool(channels_list[1], channels_list[1])
        self.Inject_p4_2 = InjectionMultiSum_Auto_pool(channels_list[1], channels_list[1])
        
        # 进一步提取和融合信息
        self.Rep_p4 = RepBlock(
                in_channels=channels_list[1],  # 256
                out_channels=channels_list[1],  # 256
                n=num_repeats[0],
                block=block
        )
        # 进一步提取和融合信息
        self.Rep_p4_2 = RepBlock(
                in_channels=channels_list[1],  # 256
                out_channels=channels_list[1],  # 256
                n=num_repeats[0],
                block=block
        )


        # 改变P4的通道数： 256 -> 128
        self.reduce_layer_p4 = SimConv(
                in_channels=channels_list[1],  # 256
                out_channels=channels_list[2],  # 128
                kernel_size=1,
                stride=1
        )
        # 改变P4的通道数： 256 -> 128
        self.reduce_layer_p4_2 = SimConv(
                in_channels=channels_list[1],  # 256
                out_channels=channels_list[2],  # 128
                kernel_size=1,
                stride=1
        )
        
        # 融合特征图 ： （C2、C3、P4） 作为F_local输入
        self.LAF_p3 = SimFusion_3in(
                in_channel_list=[channels_list[3], channels_list[2]],  # 512, 256       / 256 256
                out_channels=channels_list[2],  # 256   / 128
        )
        self.LAF_p3_2 = SimFusion_3in(
                in_channel_list=[channels_list[3], channels_list[2]],  # 512, 256       / 256 256
                out_channels=channels_list[2],  # 256   / 128
        )

        # 信息注入      将F_local 与 split后的第二个特征图融合
        self.Inject_p3 = InjectionMultiSum_Auto_pool(channels_list[2], channels_list[2])
        self.Inject_p3_2 = InjectionMultiSum_Auto_pool(channels_list[2], channels_list[2])

        # 进一步提取和融合信息
        self.Rep_p3 = RepBlock(
                in_channels=channels_list[2],  # 128
                out_channels=channels_list[2],  # 128
                n=num_repeats[1],
                block=block
        )
        self.Rep_p3_2 = RepBlock(
                in_channels=channels_list[2],  # 128
                out_channels=channels_list[2],  # 128
                n=num_repeats[1],
                block=block
        )

        # 改变P3的通道数： 256 -> 128
        self.reduce_layer_p3 = SimConv(
                in_channels=channels_list[2],  # 256
                out_channels=channels_list[3],  # 128
                kernel_size=1,
                stride=1
        )
        self.reduce_layer_p3_2 = SimConv(
                in_channels=channels_list[2],  # 256
                out_channels=channels_list[3],  # 128
                kernel_size=1,
                stride=1
        )
        # 融合特征图 ： （C2、p3） 作为F_local输入
        self.LAF_p2 = SimFusion_2in(
            in_channels = channels_list[3],
            out_channels = channels_list[3],
        )
        self.LAF_p2_2 = SimFusion_2in(
            in_channels = channels_list[3],
            out_channels = channels_list[3],
        )

        # 信息注入      将F_local 与 split后的第三个特征图融合
        self.Inject_p2 = InjectionMultiSum_Auto_pool(channels_list[3], channels_list[3])
        self.Inject_p2_2 = InjectionMultiSum_Auto_pool(channels_list[3], channels_list[3])

        # 进一步提取和融合信息
        self.Rep_p2 = RepBlock(
                in_channels=channels_list[3],  # 64
                out_channels=channels_list[3],  # 64
                n=num_repeats[2],
                block=block
        )
        self.Rep_p2_2 = RepBlock(
                in_channels=channels_list[3],  # 64
                out_channels=channels_list[3],  # 64
                n=num_repeats[2],
                block=block
        )
        
        # 特征对齐 对所有输入特征图进行相同的池化操作来达到目的（这里很奇怪，图上并没有对p5进行池化？）
        self.high_FAM = PyramidPoolAgg(stride=c2t_stride, pool_mode=pool_mode)
        self.high_FAM_2 = PyramidPoolAgg(stride=c2t_stride, pool_mode=pool_mode)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        # 信息融合      一共有block_num个块堆叠而成 MHA+FFN
        self.high_IFM = TopBasicLayer(
                block_num=depths,
                embedding_dim=embed_dim_n,
                key_dim=key_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                attn_ratio=attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
        )
        self.high_IFM_2 = TopBasicLayer(
                block_num=depths,
                embedding_dim=embed_dim_n,
                key_dim=key_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                attn_ratio=attn_ratios,
                drop=0, attn_drop=0,
                drop_path=dpr,
        )

        # 调整通道数，方便后续split
        self.conv_1x1_n = nn.Conv2d(embed_dim_n, sum(trans_channels[3:6]), 1, 1, 0)
        self.conv_1x1_n_2 = nn.Conv2d(embed_dim_n, sum(trans_channels[3:6]), 1, 1, 0)

        # 融合特征图（p2，p3）  用作F_local的输入
        self.LAF_n3 = AdvPoolFusion()
        self.LAF_n3_2 = AdvPoolFusion()

        # 信息注入 将F_local 与 split后的第一个特征图融合
        self.Inject_n3 = InjectionMultiSum_Auto_pool(channels_list[2], channels_list[2])
        self.Inject_n3_2 = InjectionMultiSum_Auto_pool(channels_list[2], channels_list[2])
        # 进一步提取和融合信息
        self.Rep_n3 = RepBlock(
                in_channels=channels_list[2],  # 128 + 128
                out_channels=channels_list[2],  # 256
                n=num_repeats[3],
                block=block
        )
        self.Rep_n3_2 = RepBlock(
                in_channels=channels_list[2],  # 128 + 128
                out_channels=channels_list[2],  # 256
                n=num_repeats[3],
                block=block
        )
        
        # 融合特征图（p3，p4）  用作F_local的输入
        self.LAF_n4 = AdvPoolFusion()
        self.LAF_n4_2 = AdvPoolFusion()

        # 信息注入 将F_local 与 split后的第一个特征图融合
        self.Inject_n4 = InjectionMultiSum_Auto_pool(channels_list[1], channels_list[1])
        self.Inject_n4_2 = InjectionMultiSum_Auto_pool(channels_list[1], channels_list[1])
        # 进一步提取和融合信息
        self.Rep_n4 = RepBlock(
                in_channels=channels_list[1],  # 128 + 128
                out_channels=channels_list[1],  # 256
                n=num_repeats[4],
                block=block
        )
        self.Rep_n4_2 = RepBlock(
                in_channels=channels_list[1],  # 128 + 128
                out_channels=channels_list[1],  # 256
                n=num_repeats[4],
                block=block
        )
        
        # 融合特征图（n4,c5） 用作F_local的输入
        self.LAF_n5 = AdvPoolFusion()
        self.LAF_n5_2 = AdvPoolFusion()

        # 信息注入      将F_local 与 split后的第二个特征图融合
        self.Inject_n5 = InjectionMultiSum_Auto_pool(channels_list[0], channels_list[0])
        self.Inject_n5_2 = InjectionMultiSum_Auto_pool(channels_list[0], channels_list[0])
        
        # 进一步提取和融合信息
        self.Rep_n5 = RepBlock(
                in_channels=channels_list[0],  # 256 + 256
                out_channels=channels_list[0],  # 512
                n=num_repeats[5],
                block=block
        )
        self.Rep_n5_2 = RepBlock(
                in_channels=channels_list[0],  # 256 + 256
                out_channels=channels_list[0],  # 512
                n=num_repeats[5],
                block=block
        )
        
        self.trans_channels = trans_channels
    
    def forward(self, input , input2):
        (c2, c3, c4, c5) = input
        (t2, t3, t4, t5) = input2

        low_align_feat = self.low_FAM(input)
        low_align_feat2 = self.low_FAM_2(input2)

        low_fuse_feat = self.low_IFM(low_align_feat)
        low_fuse_feat2 = self.low_IFM_2(low_align_feat2)

        low_global_info = low_fuse_feat.split(self.trans_channels[0:3], dim=1)
        low_global_info2 = low_fuse_feat2.split(self.trans_channels[0:3], dim=1)
        
        ## inject low-level global info to p4
        c5_half = self.reduce_layer_c5(c5)
        t5_half = self.reduce_layer_c5_2(t5)

        p4_adjacent_info = self.LAF_p4([c3, c4, c5_half])
        p4_adjacent_info2 = self.LAF_p4_2([t3, t4, t5_half])

        p4 = self.Inject_p4(p4_adjacent_info, low_global_info2[0])
        p4_2 = self.Inject_p4(p4_adjacent_info2, low_global_info[0])

        p4 = self.Rep_p4(p4)
        p4_2 = self.Rep_p4_2(p4_2)
        
        ## inject low-level global info to p3
        p4_half = self.reduce_layer_p4(p4)
        p4_half_2 = self.reduce_layer_p4_2(p4_2)

        p3_adjacent_info = self.LAF_p3([c2, c3, p4_half])
        p3_adjacent_info2 = self.LAF_p3_2([t2, t3, p4_half_2])

        p3 = self.Inject_p3(p3_adjacent_info, low_global_info2[1])
        p3_2 = self.Inject_p3_2(p3_adjacent_info2, low_global_info[1])

        p3 = self.Rep_p3(p3)
        p3_2 = self.Rep_p3_2(p3_2)

        ## inject low-level global info to p2
        p3_half = self.reduce_layer_p3(p3)
        p3_half_2 = self.reduce_layer_p3_2(p3_2)

        p2_adjacent_info = self.LAF_p2([c2,p3_half])
        p2_adjacent_info2 = self.LAF_p2_2([t2,p3_half_2])

        p2 = self.Inject_p2(p2_adjacent_info,low_global_info2[2])
        p2_2 = self.Inject_p2_2(p2_adjacent_info2,low_global_info[2])

        p2 = self.Rep_p2(p2)
        p2_2 = self.Rep_p2_2(p2_2)
        
        # High-GD
        ## use transformer fusion global info
        high_align_feat = self.high_FAM([p2, p3, p4, c5])
        high_align_feat_2 = self.high_FAM([p2_2, p3_2, p4_2, t5])

        high_fuse_feat = self.high_IFM(high_align_feat)
        high_fuse_feat_2 = self.high_IFM_2(high_align_feat_2)

        high_fuse_feat = self.conv_1x1_n(high_fuse_feat)
        high_fuse_feat_2 = self.conv_1x1_n_2(high_fuse_feat_2)

        high_global_info = high_fuse_feat.split(self.trans_channels[3:6], dim=1)
        high_global_info2 = high_fuse_feat_2.split(self.trans_channels[3:6], dim=1)
        
        ## inject low-level global info to n3
        n3_adjacent_info = self.LAF_n3(p2, p3_half)
        n3_adjacent_info2 = self.LAF_n3_2(p2_2, p3_half_2)

        n3 = self.Inject_n3(n3_adjacent_info, high_global_info2[0])
        n3_2 = self.Inject_n3_2(n3_adjacent_info2, high_global_info[0])

        n3 = self.Rep_n3(n3)
        n3_2 = self.Rep_n3_2(n3_2)

        ## inject low-level global info to n4
        n4_adjacent_info = self.LAF_n4(n3, p4_half)
        n4_adjacent_info2 = self.LAF_n4_2(n3_2, p4_half_2)

        n4 = self.Inject_n4(n4_adjacent_info, high_global_info2[1])
        n4_2 = self.Inject_n4_2(n4_adjacent_info2, high_global_info[1])

        n4 = self.Rep_n4(n4)
        n4_2 = self.Rep_n4_2(n4_2)
        
        ## inject low-level global info to n5
        n5_adjacent_info = self.LAF_n5(n4, c5_half)
        n5_adjacent_info2 = self.LAF_n5_2(n4_2, t5_half)

        n5 = self.Inject_n5(n5_adjacent_info, high_global_info2[2])
        n5_2 = self.Inject_n5_2(n5_adjacent_info2, high_global_info[2])

        n5 = self.Rep_n5(n5)
        n5_2 = self.Rep_n5(n5_2)
        
        outputs = [p2, n3, n4, n5]
        outputs2 = [p2_2, n3_2, n4_2, n5_2]
        return [outputs,outputs2]


@MODELS.register_module()
class GoldYoloNeck(nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 num_repeats=[12, 12, 12, 12],
                 ### TODO 修改参数
                 depths=2,
                 fusion_in=1920,
                 fuse_block_num=3,
                 embed_dim_p=256,
                 embed_dim_n=1408,
                 trans_channels=[256, 128, 256, 512],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 ):
        super().__init__()
        flag = len(out_channels) == 3
        in_channels.reverse()
        num_repeats = [max(3,make_round(i * deepen_factor)) for i in num_repeats]
        in_channels = [make_divisible(x , widen_factor) for x in in_channels]
        out_channels = [make_divisible(x , widen_factor) for x in out_channels]
        depths = max(1,make_round(depths * deepen_factor))
        fusion_in = sum(in_channels)
        fuse_block_num = max(1,make_round(fuse_block_num * deepen_factor))
        embed_dim_p = max(64 , make_divisible(embed_dim_p , widen_factor))
        embed_dim_n = sum(in_channels[:3]) if flag else sum(in_channels)
        trans_channels = [max(64 , make_round(x * widen_factor)) for x in trans_channels]
        if flag:
            self.neck = RepGDNeck_3(in_channels,num_repeats,depths,fusion_in,fuse_block_num,embed_dim_p,embed_dim_n,trans_channels)
        else:
            self.neck = RepGDNeck_4(in_channels,num_repeats,depths,fusion_in,fuse_block_num,embed_dim_p,embed_dim_n,trans_channels)

        self.fusion_blocks = nn.ModuleList()
        for i in range(len(out_channels)):
            self.fusion_blocks.append(
                WTConv(in_channels = out_channels[i], kernel_size=3, wt_levels=3)
            )

    def forward(self, inputs: List[torch.Tensor] , inputs2 : List[torch.Tensor]) -> tuple:
        output1 , output2 = self.neck(inputs,inputs2)

        for i in range(len(output1)):
            output1[i] = self.fusion_blocks[i](output1[i], output2[i])
        return tuple(output1)