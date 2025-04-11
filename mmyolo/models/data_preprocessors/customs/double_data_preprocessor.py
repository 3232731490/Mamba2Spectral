# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
import math
import random
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.logging import MessageHub
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of
from torch import Tensor

from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.models.utils import unfold_wo_center
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import ConfigType

try:
    import skimage
except ImportError:
    skimage = None


from mmengine.model.utils import stack_batch

from mmyolo.registry import MODELS

@MODELS.register_module()
class DualInputDetDataPreprocessor(DetDataPreprocessor):
    """Rewrite collate_fn to get faster training speed.

    Note: It must be used together with `mmyolo.datasets.utils.yolov5_collate`
    """

    def __init__(self, *args, non_blocking: Optional[bool] = True, **kwargs):
        super().__init__(*args, non_blocking=non_blocking, **kwargs)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``DetDataPreprocessorr``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            cp_data = deepcopy(data)
            cp_data['inputs'] = data['inputs2']
            out1 = super().forward(data,training)
            out2 = super().forward(cp_data,training)
            # # print(data)
            # img_path = data['data_samples'][0].img_path
            # filename = os.path.basename(img_path)
            # test_img(data['inputs'][0],data['data_samples'][0].gt_instances.bboxes.tensor,f'ori_rgb_{filename}')
            # test_img(data['inputs2'][0],data['data_samples'][0].gt_instances.bboxes.tensor,f'ori_tir_{filename}')
            # print("data",data)
            # print("==========================================")
            # print("out1",out1)
            # test_img(out1['inputs'][0],out1['data_samples'][0].gt_instances.bboxes,f'rgb_{filename}')
            # # print("==========================================")
            # # print("out2",out2)
            # test_img(out2['inputs'][0],out2['data_samples'][0].gt_instances.bboxes,f'tir_{filename}')
            # print(out)
            return {'inputs':out1['inputs'],'inputs2':out2['inputs'],'data_samples':out1['data_samples']}
        if training:
            data = self.cast_data(data)
            inputs, inputs2, data_samples = data['inputs'], data['inputs2'], data['data_samples']
            assert isinstance(data['data_samples'], dict)

            # TODO: Supports multi-scale training
            if self._channel_conversion and inputs.shape[1] == 3:
                inputs = inputs[:, [2, 1, 0], ...]
                inputs2 = inputs2[:, [2, 1, 0], ...]
            if self._enable_normalize:
                inputs = (inputs - self.mean) / self.std
                inputs2 = (inputs2 - self.mean) / self.std

            if self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    inputs, data_samples = batch_aug(inputs, data_samples)
                    inputs2, _ = batch_aug(inputs, data_samples)

            img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
            data_samples_output = {
                'bboxes_labels': data_samples['bboxes_labels'],
                'img_metas': img_metas
            }
            if 'masks' in data_samples:
                data_samples_output['masks'] = data_samples['masks']
            if 'keypoints' in data_samples:
                data_samples_output['keypoints'] = data_samples['keypoints']
                data_samples_output['keypoints_visible'] = data_samples[
                    'keypoints_visible']

            return {'inputs': inputs,'inputs2':inputs2, 'data_samples': data_samples_output}

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def test_img(inputs: torch.Tensor,label:torch.Tensor,filename:str):
    # 将inputs张量转换为numpy数组
    inputs = inputs.cpu().numpy()

    # 选择一个通道的图像进行绘制
    # 如果输入是彩色图像，则需要转置其维度
    image = np.transpose(inputs, (1, 2, 0))  # 选择第一个输入图像

    # 创建图像
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 获取bboxes
    bboxes = label.cpu().numpy()

    # 绘制每个边界框
    for bbox in bboxes:
        x, y, x2, y2 = bbox
        width = x2 - x
        height = y2 - y
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # 隐藏坐标轴
    ax.axis('off')

    # 保存图像
    fig.savefig(f'/data/nl/mmyolo2spectral/mmyolo/results_imgs/{filename}.jpg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"图像已保存为 {filename}")