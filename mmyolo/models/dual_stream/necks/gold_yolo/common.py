import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

__all__ = ('SimConv','AdvPoolFusion','SimFusion_3in','SimFusion_4in')

class SimConv(nn.Module):
    '''Normal Conv with ReLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x

# 用于融合两个特征图    最终分辨率以较小的分辨率特征图为准
class AdvPoolFusion(nn.Module):
    def forward(self, x1, x2):
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d
        
        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)
        
        return torch.cat([x1, x2], 1)
    
class SimFusion_2in(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = SimConv(in_channels, out_channels, 1, 1)
        self.cv_fuse = SimConv(out_channels * 2, out_channels, 1, 1)
    
    def forward(self, x):
        N, C, H, W = x[0].shape
        x1 = self.cv1(x[0])
        x2 = F.interpolate(x[1], size=(H, W), mode='bilinear', align_corners=False)
        return self.cv_fuse(torch.cat((x1, x2), dim=1))

# 用于融合三个特征图    最终分辨率以中间大小的分辨率特征图为准
# 注意这里要保证第三个特征图的通道数已经调整为输出通道数了；因为这个融合函数没有对第三个特征图的通道数进行调整
class SimFusion_3in(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        self.cv1 = SimConv(in_channel_list[0], out_channels, 1, 1)
        self.cv2 = SimConv(in_channel_list[1], out_channels, 1, 1)
        self.cv_fuse = SimConv(out_channels * 3, out_channels, 1, 1)
        self.downsample = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)
        self.downsample = nn.AdaptiveAvgPool2d(output_size)
        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])
        x0 = self.cv1(self.downsample(x[0]))
        x1 = self.cv2(x[1])
        x2 = F.interpolate(x[2], size=(H, W), mode='bilinear', align_corners=False)
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))


# 用于融合四个特征图    最终分辨率以第二小的特征图为准
class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
    
    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])
        
        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d
        
        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode='bilinear', align_corners=False)
        
        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out
