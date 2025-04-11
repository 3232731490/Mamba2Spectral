import copy
from typing import Callable, Dict, List, Optional, Union

import cv2
import numpy as np
from mmcv.transforms import BaseTransform, Compose
from mmcv.transforms.utils import cache_random_params, cache_randomness

from mmyolo.registry import TRANSFORMS

@TRANSFORMS.register_module()
class Image2Broadcaster(BaseTransform):
    """A transform wrapper to apply the wrapped transforms to process both
    `gt_bboxes` and `proposals` without adding any codes. It will do the
    following steps:

        1. Scatter the broadcasting targets to a list of inputs of the wrapped
           transforms. The type of the list should be list[dict, dict], which
           the first is the original inputs, the second is the processing
           results that `gt_bboxes` being rewritten by the `proposals`.
        2. Apply ``self.transforms``, with same random parameters, which is
           sharing with a context manager. The type of the outputs is a
           list[dict, dict].
        3. Gather the outputs, update the `proposals` in the first item of
           the outputs with the `gt_bboxes` in the second .

    Args:
         transforms (list, optional): Sequence of transform
            object or config dict to be wrapped. Defaults to [].

    Note: The `TransformBroadcaster` in MMCV can achieve the same operation as
          `ProposalBroadcaster`, but need to set more complex parameters.

    Examples:
        >>> pipeline = [
        >>>     dict(type='LoadImageFromFile'),
        >>>     dict(type='LoadProposals', num_max_proposals=2000),
        >>>     dict(type='LoadAnnotations', with_bbox=True),
        >>>     dict(
        >>>         type='ProposalBroadcaster',
        >>>         transforms=[
        >>>             dict(type='Resize', scale=(1333, 800),
        >>>                  keep_ratio=True),
        >>>             dict(type='RandomFlip', prob=0.5),
        >>>         ]),
        >>>     dict(type='PackDetInputs')]
    """

    def __init__(self, transforms: List[Union[dict, Callable]] = []) -> None:
        self.transforms = Compose(transforms)

    def transform(self, results: dict) -> dict:
        """Apply wrapped transform functions to process both `gt_bboxes` and
        `proposals`.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        assert results.get('img2', None) is not None, \
            '`proposals` should be in the results, please delete ' \
            '`ProposalBroadcaster` in your configs, or check whether ' \
            'you have load proposals successfully.'

        inputs = self._process_input(results)
        outputs = self._apply_transforms(inputs)
        outputs = self._process_output(outputs)
        return outputs

    def _process_input(self, data: dict) -> list:
        """Scatter the broadcasting targets to a list of inputs of the wrapped
        transforms.

        Args:
            data (dict): The original input data.

        Returns:
            list[dict]: A list of input data.
        """
        cp_data = copy.deepcopy(data)
        cp_data['img'] = cp_data['img2']
        cp_data['img_shape'] = cp_data['img_shape2']
        cp_data['img_path'] = cp_data['img_path2']
        cp_data['ori_shape'] = cp_data['ori_shape2']
        scatters = [data, cp_data]
        return scatters

    def _apply_transforms(self, inputs: list) -> list:
        """Apply ``self.transforms``.

        Args:
            inputs (list[dict, dict]): list of input data.

        Returns:
            list[dict]: The output of the wrapped pipeline.
        """
        assert len(inputs) == 2
        ctx = cache_random_params
        with ctx(self.transforms):
            output_scatters = [self.transforms(_input) for _input in inputs]
        return output_scatters

    def _process_output(self, output_scatters: list) -> dict:
        """Gathering and renaming data items.

        Args:
            output_scatters (list[dict, dict]): The output of the wrapped
                pipeline.

        Returns:
            dict: Updated result dict.
        """
        assert isinstance(output_scatters, list) and \
               isinstance(output_scatters[0], dict) and \
               len(output_scatters) == 2
        outputs = output_scatters[0]
        outputs['img2'] = output_scatters[1]['img']
        outputs['img_path2'] = output_scatters[1]['img_path']
        outputs['img_shape2'] = output_scatters[1]['img_shape']
        outputs['ori_shape2'] = output_scatters[1]['ori_shape']
        return outputs


@TRANSFORMS.register_module()
class Branch(BaseTransform):
    """A transform wrapper to apply the wrapped transforms to process both
    `gt_bboxes` and `proposals` without adding any codes. It will do the
    following steps:

        1. Scatter the broadcasting targets to a list of inputs of the wrapped
           transforms. The type of the list should be list[dict, dict], which
           the first is the original inputs, the second is the processing
           results that `gt_bboxes` being rewritten by the `proposals`.
        2. Apply ``self.transforms``, with same random parameters, which is
           sharing with a context manager. The type of the outputs is a
           list[dict, dict].
        3. Gather the outputs, update the `proposals` in the first item of
           the outputs with the `gt_bboxes` in the second .

    Args:
         transforms (list, optional): Sequence of transform
            object or config dict to be wrapped. Defaults to [].

    Note: The `TransformBroadcaster` in MMCV can achieve the same operation as
          `ProposalBroadcaster`, but need to set more complex parameters.

    Examples:
        >>> pipeline = [
        >>>     dict(type='LoadImageFromFile'),
        >>>     dict(type='LoadProposals', num_max_proposals=2000),
        >>>     dict(type='LoadAnnotations', with_bbox=True),
        >>>     dict(
        >>>         type='ProposalBroadcaster',
        >>>         transforms=[
        >>>             dict(type='Resize', scale=(1333, 800),
        >>>                  keep_ratio=True),
        >>>             dict(type='RandomFlip', prob=0.5),
        >>>         ]),
        >>>     dict(type='PackDetInputs')]
    """

    def __init__(self, transforms: List[Union[dict, Callable]] = []) -> None:
        self.transforms = Compose(transforms)

    def transform(self, results: dict) -> dict:
        """Apply wrapped transform functions to process both `gt_bboxes` and
        `proposals`.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        assert results.get('img2', None) is not None, \
            '`proposals` should be in the results, please delete ' \
            '`ProposalBroadcaster` in your configs, or check whether ' \
            'you have load proposals successfully.'

        inputs = self._process_input(results)
        outputs = self._apply_transforms(inputs)
        outputs = self._process_output(outputs)
        return outputs

    def _process_input(self, data: dict) -> list:
        """Scatter the broadcasting targets to a list of inputs of the wrapped
        transforms.

        Args:
            data (dict): The original input data.

        Returns:
            list[dict]: A list of input data.
        """
        
        cp_data = copy.deepcopy(data)
        cp_data['img'] = cp_data['img2']
        cp_data['img_shape'] = cp_data['img_shape2']
        cp_data['img_path'] = cp_data['img_path2']
        cp_data['ori_shape'] = cp_data['ori_shape2']
        scatters = [data, cp_data]
        return scatters

    def _apply_transforms(self, inputs: list) -> list:
        """Apply ``self.transforms``.

        Args:
            inputs (list[dict, dict]): list of input data.

        Returns:
            list[dict]: The output of the wrapped pipeline.
        """
        assert len(inputs) == 2
        output_scatters = [self.transforms(_input) for _input in inputs]
        return output_scatters

    def _process_output(self, output_scatters: list) -> dict:
        """Gathering and renaming data items.

        Args:
            output_scatters (list[dict, dict]): The output of the wrapped
                pipeline.

        Returns:
            dict: Updated result dict.
        """
        assert isinstance(output_scatters, list) and \
               isinstance(output_scatters[0], dict) and \
               len(output_scatters) == 2
        outputs = output_scatters[0]
        outputs['img2'] = output_scatters[1]['img']
        outputs['img_path2'] = output_scatters[1]['img_path']
        outputs['img_shape2'] = output_scatters[1]['img_shape']
        outputs['ori_shape2'] = output_scatters[1]['ori_shape']

        return outputs