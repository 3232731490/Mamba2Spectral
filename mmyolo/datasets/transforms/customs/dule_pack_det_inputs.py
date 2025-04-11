# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import cv2
import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData

from mmdet.structures import DetDataSample, ReIDDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes

from mmyolo.registry import TRANSFORMS

@TRANSFORMS.register_module()
class DoublePackDetInputs(BaseTransform):
    """Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction', 'pad_ltrb')):
        if 'pad_ltrb' not in meta_keys:
            meta_keys = meta_keys + ('pad_ltrb', )
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results and 'img2' in results:
            img = results['img']
            img2 = results['img2']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if len(img2.shape) < 3:
                img2 = np.expand_dims(img2, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            if not img2.flags.c_contiguous:
                img2 = np.ascontiguousarray(img2.transpose(2, 0, 1))
                img2 = to_tensor(img2)
            else:
                img2 = to_tensor(img2).permute(2, 0, 1).contiguous()

            # print(img.shape, img2.shape)
            # cv2.imwrite("debug1.jpg", img.permute((1,2,0)).cpu().numpy())
            # cv2.imwrite("debug2.jpg", img2.permute((1,2,0)).cpu().numpy())
            # raise RuntimeError()
            packed_results['inputs'], packed_results['inputs2'] = img, img2

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        # # 从img_path读取文件名
        # img_path = data_sample.img_path
        # filename = os.path.basename(img_path)
        # test_img(img,data_sample.gt_instances.bboxes.tensor,f'rgb_{filename}')
        # test_img(img2,data_sample.gt_instances.bboxes.tensor,f'tir_{filename}')
        # print(packed_results)
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def test_img(inputs: torch.Tensor,label:torch.Tensor,filename:str):
    print(inputs.shape)
    # 将inputs张量转换为numpy数组
    inputs = inputs.numpy()

    # 选择一个通道的图像进行绘制
    # 如果输入是彩色图像，则需要转置其维度
    image = np.transpose(inputs, (1, 2, 0))  # 选择第一个输入图像

    # 创建图像
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 获取bboxes
    bboxes = label.numpy()

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