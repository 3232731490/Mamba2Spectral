from typing import List, Sequence

import numpy as np
import torch
from mmengine.dataset import COLLATE_FUNCTIONS
from mmengine.dist import get_dist_info

from mmyolo.registry import TASK_UTILS
import cv2

@COLLATE_FUNCTIONS.register_module()
def dual_yolo_collate(data_batch: Sequence,
                   use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    batch_imgs1 = []
    batch_imgs2 = []

    batch_bboxes_labels = []
    batch_masks = []
    batch_keyponits = []
    batch_keypoints_visible = []


    for i in range(len(data_batch)):
        datasamples = data_batch[i]['data_samples']
        inputs1 = data_batch[i]['inputs']
        inputs2 = data_batch[i]['inputs2']
        batch_imgs1.append(inputs1)
        batch_imgs2.append(inputs2)
        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        gt_labels = datasamples.gt_instances.labels
        if 'masks' in datasamples.gt_instances:
            masks = datasamples.gt_instances.masks
            batch_masks.append(masks)
        if 'gt_panoptic_seg' in datasamples:
            batch_masks.append(datasamples.gt_panoptic_seg.pan_seg)
        if 'keypoints' in datasamples.gt_instances:
            keypoints = datasamples.gt_instances.keypoints
            keypoints_visible = datasamples.gt_instances.keypoints_visible
            batch_keyponits.append(keypoints)
            batch_keypoints_visible.append(keypoints_visible)

        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes),
                                  dim=1)
        batch_bboxes_labels.append(bboxes_labels)
    collated_results = {
        'data_samples': {
            'bboxes_labels': torch.cat(batch_bboxes_labels, 0)
        }
    }
    if len(batch_masks) > 0:
        collated_results['data_samples']['masks'] = torch.cat(batch_masks, 0)

    if len(batch_keyponits) > 0:
        collated_results['data_samples']['keypoints'] = torch.cat(
            batch_keyponits, 0)
        collated_results['data_samples']['keypoints_visible'] = torch.cat(
            batch_keypoints_visible, 0)

    if use_ms_training:
        collated_results['inputs'] = batch_imgs1
        collated_results['inputs2'] = batch_imgs2
    else:
        collated_results['inputs'] = torch.stack(batch_imgs1, 0)
        collated_results['inputs2'] = torch.stack(batch_imgs2, 0)
    return collated_results
