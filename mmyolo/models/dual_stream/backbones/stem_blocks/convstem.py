from mmcv.cnn import ConvModule

from mmyolo.registry import MODELS
from mmengine.model import BaseModule
from mmyolo.models.utils import make_divisible, make_round

@MODELS.register_module()
class ConvStem(BaseModule):
    """ConvStem模块,用于YOLOv8模型的stem部分。

    Args:
        widen_factor (float): 通道宽度倍率,默认为1.0。
        deepen_factor (float): 深度倍率,默认为1.0。
        in_channels (int): 输入通道数,默认为3。
        out_channels (int): 输出通道数,默认为64。
        kernel_size (int): 卷积核大小,默认为3。
        stride (int): 步幅,默认为2。
        padding (int): 填充大小,默认为1。
        norm_cfg: norm层配置,默认为None,
        act_cfg: act层配置,默认为None,
    """

    def __init__(self,
                 widen_factor: float = 1.0,
                 in_channels: int = 3,
                 out_channels: int = 64,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 norm_cfg: dict = None,
                 act_cfg: dict = None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = make_divisible(out_channels, widen_factor)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.conv = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def forward(self, x):
        """前向传播函数。
        """
        return self.conv(x)