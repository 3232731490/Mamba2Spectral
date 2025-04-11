# Copyright (c) OpenMMLab. All rights reserved.
import torch
import copy
from typing import List, Tuple, Union
from mmdet.models.detectors.base import BaseDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmengine.dist import get_world_size
from mmengine.logging import print_log


from torch import Tensor

from mmyolo.registry import MODELS

@MODELS.register_module()
class YOLODualNeckDetector(BaseDetector):
    r"""Implementation of YOLO Series

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of YOLO. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of YOLO. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
        use_syncbn (bool): whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 pre_handle: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        
        self.pre_handle = None
        if pre_handle is not None:
            self.pre_handle = MODELS.build(pre_handle)

        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        # # Find backbone parameters
        # copy_ori = False
        # ori_backbone_params = []
        # ori_backbone_key = []
        # for k, v in state_dict.items():
        #     if (k.startswith("backbone") and "backbone1" not in k and "backbone2" not in k):
        #         # or (k.startswith("neck")  and "neck1" not in k and "neck2" not in k):
        #         # Pretrained on original model
        #         ori_backbone_params += [v]
        #         ori_backbone_key += [k]
        #         copy_ori = True

        # if copy_ori:
        #     for k, v in zip(ori_backbone_key, ori_backbone_params):
        #         state_dict[k.replace("backbone", "backbone1")] = v
        #         state_dict[k.replace("backbone", "backbone2")] = copy.deepcopy(v)
        #         # state_dict[k.replace("neck", "neck1")] = v
        #         # state_dict[k.replace("neck", "neck2")] = copy.deepcopy(v)
        #         del state_dict[k]
        #     # Force set the strict to "False"
        #     strict = False

        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        res = super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
        return res

    def forward(self,
                inputs: torch.Tensor,
                inputs2: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor'):

        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """

        # from mmdet.visualization.local_visualizer import DetLocalVisualizer
        # dv = DetLocalVisualizer()
        # image = inputs2.permute(0, 2,3,1)[0].cpu().numpy()[:,:,::-1] * 255
        # image2 = inputs.permute(0, 2,3,1)[0].cpu().numpy()[:,:,::-1] * 255
        # dv.add_datasample('image', image, data_samples[0], draw_gt=True, show=True)
        # dv.add_datasample('image2', image2, data_samples[0], draw_gt=True, show=True)

        if mode == 'loss':
            return self.loss(inputs, inputs2, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, inputs2, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, inputs2, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def extract_feat(self, batch_inputs: Tensor, batch_inputs2: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        if self.pre_handle is not None:
            batch_inputs = self.pre_handle(batch_inputs)

        z = self.backbone(batch_inputs,batch_inputs2)

        if self.with_neck:
            z = self.neck(z)
            
        return z

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_inputs2: Tensor,
                 batch_data_samples: OptSampleList = None):
        x = self.extract_feat(batch_inputs,batch_inputs2)
        results_list = self.bbox_head.forward(x)
        return results_list

    def loss(self, batch_inputs: Tensor, batch_inputs2: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:

        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs,batch_inputs2)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor, batch_inputs2: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x = self.extract_feat(batch_inputs,batch_inputs2)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples