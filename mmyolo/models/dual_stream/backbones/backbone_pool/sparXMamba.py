from mmcv.cnn import ConvModule

from mmyolo.models.utils import make_divisible, make_round
from .sparXBackbone import SparXMambaStage
from copy import deepcopy

def build_sparXMambaStage(setting: list):
    """Build a stage layer.
    Args:
        stage_idx (int): The index of a stage layer.
        setting (list): The architecture setting of a stage layer.
    """
    return [SparXMambaStage(
        in_channels=setting.in_channels,
        out_channels=setting.out_channels,
        patch_embed=setting.patch_embed,
        depth=setting.depth,
        sr_ratio=setting.sr_ratio,
        is_first_stage=setting.is_first_stage,
        max_dense_depth=setting.max_dense_depth,
        dense_step=setting.dense_step,
        dense_start=setting.dense_start,
        widen_factor=setting.widen_factor,
        deepen_factor=setting.deepen_factor
    )]