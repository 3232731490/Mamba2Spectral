from mmcv.cnn import ConvModule

from mmyolo.models.layers import CSPLayerWithTwoConv, SPPFBottleneck
from mmyolo.models.utils import make_divisible, make_round


def build_yoloV8CSPDarknetStage(setting: list):
    """Build a stage layer.

    Args:
        stage_idx (int): The index of a stage layer.
        setting (list): The architecture setting of a stage layer.
    """
    in_channels, out_channels, num_blocks, add_identity, use_spp = setting.in_channels , setting.out_channels , setting.num_blocks , setting.add_identity , setting.use_spp
    in_channels = make_divisible(in_channels, setting.widen_factor)
    out_channels = make_divisible(out_channels, setting.widen_factor)
    num_blocks = make_round(num_blocks, setting.deepen_factor)
    stage = []
    conv_layer = ConvModule(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        norm_cfg=setting.norm_cfg,
        act_cfg=setting.act_cfg)
    stage.append(conv_layer)
    csp_layer = CSPLayerWithTwoConv(
        out_channels,
        out_channels,
        num_blocks=num_blocks,
        add_identity=add_identity,
        norm_cfg=setting.norm_cfg,
        act_cfg=setting.act_cfg)
    stage.append(csp_layer)
    if use_spp:
        spp = SPPFBottleneck(
            out_channels,
            out_channels,
            kernel_sizes=5,
            norm_cfg=setting.norm_cfg,
            act_cfg=setting.act_cfg)
        stage.append(spp)
    return stage
