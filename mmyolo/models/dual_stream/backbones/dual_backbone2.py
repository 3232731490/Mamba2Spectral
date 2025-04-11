from typing import List, Tuple, Union
from abc import ABCMeta
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.backbones.csp_darknet import CSPLayer, Focus
from mmdet.utils import ConfigType, OptMultiConfig

from mmengine.model import BaseModule

from mmyolo.registry import MODELS
from mmyolo.models.layers import CSPLayerWithTwoConv, SPPFBottleneck
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.models.backbones import BaseBackbone

from copy import deepcopy

@MODELS.register_module()
class Dual_YOLOv8CSPDarknet2(BaseModule, metaclass=ABCMeta):
    """CSP-Darknet backbone used in YOLOv8.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5}.
            Defaults to P5.
        last_stage_out_channels (int): Final layer output channel.
            Defaults to 1024.
        plugins (list[dict]): List of plugins for stages, each dict contains:
            - cfg (dict, required): Cfg dict to build plugin.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        input_channels (int): Number of input image channels. Defaults to: 3.
        out_indices (Tuple[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        init_cfg (Union[dict,list[dict]], optional): Initialization config
            dict. Defaults to None.

    Example:
        >>> from mmyolo.models import YOLOv8CSPDarknet
        >>> import torch
        >>> model = YOLOv8CSPDarknet()
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = model(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    # the final out_channels will be set according to the param.
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, None, 3, True, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 6, True, False],
               [256, 512, 6, True, False], [512, 768, 3, True, False],
               [768, None, 3,False, True]]
    }

    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 fusion_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None,
                 fusion_block:ConfigType = dict(
                    type='BaseFusion',
                    in_channels = [256 , 512 , 1024],
                ),
                fusion_module:ConfigType = dict(
                    type='Add'
                )):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        super().__init__(init_cfg)

        self.arch_setting = self.arch_settings[arch]
        self.input_channels = input_channels
        self.out_indices = out_indices
        self.fusion_indices = fusion_indices
        self.frozen_stages = frozen_stages
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.plugins = plugins

        self.stem_1 = self.build_stem_layer()
        self.stem_2 = self.build_stem_layer()

        self.layers = ['stem']
        self.fusion_block = fusion_block
        self.fusion_block['in_channels'] = [make_divisible(x, self.widen_factor) for x in self.fusion_block['in_channels']]
        self.fusion_module = fusion_module
        self.fusion_module2 = deepcopy(fusion_module)
        if 'in_channels' in self.fusion_module:
            self.fusion_module['in_channels'] = [make_divisible(x, self.widen_factor) for x in self.fusion_module['in_channels']]
            self.fusion_module2['in_channels'] = [make_divisible(x, self.widen_factor) * 2 for x in self.fusion_module2['in_channels']]
        i = 0
        i1 = 0
        fusion_block_configs = self.generate_configs(self.fusion_block)
        fusion_module_configs = self.generate_configs(self.fusion_module)
        fusion_module_configs2 = self.generate_configs(self.fusion_module2)
        if len(fusion_module_configs) == 1:
            fusion_module_configs = [fusion_module_configs for _ in fusion_block_configs]
            fusion_module_configs2 = [fusion_module_configs2 for _ in fusion_block_configs]
        for idx, setting in enumerate(self.arch_settings[arch]):
            stage = []
            stage2 = []
            stage += self.build_stage_layer(idx, setting)
            stage2 += self.build_stage_layer(idx, setting)
            if plugins is not None:
                stage += self.make_stage_plugins(plugins, idx, setting)
                stage2 += self.make_stage_plugins(plugins, idx, setting)
            self.add_module(f'stage{idx + 1}_1', nn.Sequential(*stage))
            self.add_module(f'stage{idx + 1}_2', nn.Sequential(*stage2))
            self.layers.append(f'stage{idx + 1}')
            if idx + 1 in fusion_indices:
                self.add_module(f'fusion_block{idx+1}',self.build_fusion_layer(fusion_block_configs[i1]))
                self.add_module(f'fusion_module{idx+1}_1',self.build_fusion_module_layer(fusion_module_configs[i]))
                i1+=1
            if idx + 1 in out_indices:
                self.add_module(f'fusion_module{idx+1}_2',self.build_fusion_module_layer(fusion_module_configs2[i]))
                i+=1

    def generate_configs(self,base_config):
        # 找到所有需要展开的列表
        keys_with_lists = {k: v for k, v in base_config.items() if isinstance(v, list)}
        if not keys_with_lists:
            return base_config
        # 获取列表的长度，假设所有列表的长度相同
        num_configs = len(next(iter(keys_with_lists.values())))

        # 生成多个配置
        configs = []
        for i in range(num_configs):
            new_config = deepcopy(base_config)
            for k, v in keys_with_lists.items():
                new_config[k] = v[i]  # 取出每个列表的第i个值
            configs.append(new_config)
        
        return configs

    def build_fusion_layer(self,params:ConfigType) -> nn.Module:
        return MODELS.build(params)
    
    def build_fusion_module_layer(self,params:ConfigType) -> nn.Module:
        return MODELS.build(params)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting
        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer = CSPLayerWithTwoConv(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(csp_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage

    def init_weights(self):
        """Initialize the parameters."""
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()

    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        outs = []
        for i, layer_name in enumerate(self.layers):
            if i in self.fusion_indices:
                fusion_block = getattr(self,f'fusion_block{i}')
                fusion_module = getattr(self,f'fusion_module{i}_1')
                f_out = fusion_block(inputs1,inputs2)
                inputs1 = fusion_module(f_out[0],inputs1)
                inputs2 = fusion_module(f_out[-1],inputs2)
            layer = getattr(self, f'{layer_name}_1')
            layer2 = getattr(self, f'{layer_name}_2')
            inputs1 = layer(inputs1)
            inputs2 = layer2(inputs2)
            if i in self.out_indices:
                fusion_module = getattr(self,f'fusion_module{i}_2')
                out = fusion_module(inputs1,inputs2)
                outs.append(out)
        return tuple(outs)
