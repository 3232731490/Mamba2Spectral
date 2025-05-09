from typing import List, Tuple, Union
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmyolo.registry import MODELS
from mmyolo.models.utils import make_divisible, make_round
from copy import deepcopy
from .backbone_pool.all_backbones import *
from mmcv.cnn import ConvModule


@MODELS.register_module()
class GeneralDualBackbone(BaseModule):
    """通用的双流Backbone模块,支持自定义每个stage的模块类型和配置。

    Args:
        stages (List[dict]): 每个stage的配置列表,每个配置包含：
            - type (str): 模块类型。
            - settings (dict): 模块参数。
        fusion_block (dict): 融合模块的配置。
        fusion_module (dict): 融合操作模块的配置。
        input_channels (int): 输入通道数,默认为3。
        stem_out_channels: stem层输出通道数,默认为64。,
        norm_cfg: norm层配置,默认为None,
        act_cfg: act层配置,默认为None,
        out_indices (Tuple[int]): 输出的stage索引,默认为(2, 3, 4)。
        fusion_indices (Tuple[int]): 融合的stage索引,默认为(2, 3, 4)。
        widen_factor (float): 通道宽度倍率,默认为1.0。
        deepen_factor (float): 深度倍率,默认为1.0。
        init_cfg (dict, optional): 初始化配置,默认为None。
    """

    def __init__(self,
                 stages: dict,
                 stem_block: dict,
                 fusion_block: dict,
                 fusion_module: dict,
                 head_input_module : dict,
                 input_channels: int = 3,
                 norm_cfg: dict = None,
                 act_cfg: dict = None,
                 fusion_flag : bool = True,
                 out_indices: Tuple[int] = (2, 3, 4),
                 fusion_indices: Tuple[int] = (2, 3, 4),
                 widen_factor: float = 1.0,
                 deepen_factor: float = 1.0,
                 init_cfg: dict = None,):
        super().__init__(init_cfg)

        self.stages = stages
        self.fusion_block = fusion_block
        self.fusion_module = fusion_module
        self.input_channels = input_channels
        self.out_indices = out_indices
        self.fusion_indices = fusion_indices
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.head_input_module = head_input_module
        self.fusion_flag = fusion_flag

        self.layers = []
        self.fusion_block['in_channels'] = [make_divisible(x, self.widen_factor) for x in self.fusion_block['in_channels']]
        if 'in_channels' in self.fusion_module:
            self.fusion_module['in_channels'] = [make_divisible(x, self.widen_factor) for x in self.fusion_module['in_channels']]
            self.head_input_module['in_channels'] = [make_divisible(x, self.widen_factor) for x in self.head_input_module['in_channels']]

        self.stem_1 = self.build_stem_layer(stem_block)
        self.stem_2 = self.build_stem_layer(stem_block)
        self.layers.append('stem')

        fusion_block_configs = self.generate_configs(self.fusion_block)
        fusion_module_configs = self.generate_configs(self.fusion_module)
        head_input_configs = self.generate_configs(self.head_input_module)
        stages_configs = self.generate_configs(self.stages.settings)
        
        self.build_stage_function = all_backbones[self.stages.type]
        
        if len(fusion_module_configs) == 1:
            fusion_module_configs = [fusion_module_configs for _ in fusion_block_configs]
            head_input_configs = [head_input_configs for _ in fusion_block_configs]

        i,i1 = 0 , 0
        for idx, stage_cfg in enumerate(stages_configs):
            stage1 = self.build_stage_function(stage_cfg)
            stage2 = self.build_stage_function(stage_cfg)

            self.add_module(f'stage{idx + 1}_1', nn.Sequential(*stage1))
            self.add_module(f'stage{idx + 1}_2', nn.Sequential(*stage2))
            # print(f'stage{idx + 1}_1', nn.Sequential(*stage1))
            self.layers.append(f'stage{idx + 1}')

            if idx + 1 in self.fusion_indices:
                self.add_module(f'fusion_block{idx + 1}', self.build_fusion_layer(fusion_block_configs[i1]))
                self.add_module(f'fusion_module{idx + 1}', self.build_fusion_module_layer(fusion_module_configs[i]))
                i1+=1
                if not self.fusion_flag:
                    i+=1
            if self.fusion_flag and idx + 1 in self.out_indices:
                self.add_module(f'head_input_module{idx + 1}', self.build_fusion_module_layer(head_input_configs[i]))
                i+=1

    def generate_configs(self, base_config):
        """生成配置列表,用于支持多种配置展开。"""
        keys_with_lists = {k: v for k, v in base_config.items() if isinstance(v, list)}
        if not keys_with_lists:
            return [base_config]
        num_configs = len(next(iter(keys_with_lists.values())))
        configs = []
        for i in range(num_configs):
            new_config = deepcopy(base_config)
            for k, v in keys_with_lists.items():
                new_config[k] = v[i]
            configs.append(new_config)
        return configs

    def build_fusion_layer(self, params: dict) -> nn.Module:
        """构建融合模块。"""
        return MODELS.build(params)

    def build_fusion_module_layer(self, params: dict) -> nn.Module:
        """构建融合操作模块。"""
        return MODELS.build(params)

    def build_stem_layer(self , params : dict) -> nn.Module:
        """构建stem层。"""
        return MODELS.build(params)

    def forward(self, inputs1: torch.Tensor, inputs2: torch.Tensor) -> tuple:
        """前向传播。"""
        if self.fusion_flag:
            outs = []
        else:
            outs = [[],[]]
        for i, layer_name in enumerate(self.layers):
            if i in self.fusion_indices:
                fusion_block = getattr(self, f'fusion_block{i}')
                fusion_module = getattr(self, f'fusion_module{i}')
                f_out = fusion_block(inputs1, inputs2)
                inputs1 = fusion_module(f_out[0], inputs1)
                inputs2 = fusion_module(f_out[-1], inputs2)
            layer1 = getattr(self, f'{layer_name}_1')
            layer2 = getattr(self, f'{layer_name}_2')
            inputs1 = layer1(inputs1)
            inputs2 = layer2(inputs2)
            if i in self.out_indices:
                if self.fusion_flag:
                    fusion_module = getattr(self, f'head_input_module{i}')
                    out = fusion_module(inputs1, inputs2)
                    outs.append(out)
                else:
                    outs[0].append(inputs1)
                    outs[1].append(inputs2)
        return tuple(outs)
