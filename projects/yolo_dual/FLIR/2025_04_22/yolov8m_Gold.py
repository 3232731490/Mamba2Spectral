_base_ = '/data/nl/Mamba2Spectral/projects/yolo_dual/datasets/FILR.py'

_base_.strides = [4, 8, 16, 32]
model = dict(
        type='YOLODualMidFusionDetector',
    data_preprocessor=dict(
        type='DualInputDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        out_indices=(1,2, 3, 4),
        last_stage_out_channels=_base_.last_stage_out_channels,
        deepen_factor=_base_.deepen_factor,
        widen_factor=_base_.widen_factor,
        norm_cfg=_base_.norm_cfg,

        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        _delete_=True,
        type='GoldYoloNeck',
        deepen_factor=_base_.deepen_factor,
        widen_factor=_base_.widen_factor,
        in_channels = [128,256,512,_base_.last_stage_out_channels],
        out_channels = [128, 256,512,_base_.last_stage_out_channels],
        num_repeats=[12, 12, 12, 12,12, 12],
        depths=2,
        fusion_in=1920,
        fuse_block_num=3,
        embed_dim_p=256,
        embed_dim_n=1408,
        # trans_channels=[512, 256, 512, 1024],   # out 3
        trans_channels=[512, 256,128,256, 512, 1024],   # out 4
    ),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=_base_.num_classes,
            in_channels=[128,256, 512, _base_.last_stage_out_channels],
            widen_factor=_base_.widen_factor,
            reg_max=16,
            norm_cfg=_base_.norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=_base_.strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=_base_.strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=_base_.loss_cls_weight),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=_base_.loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=_base_.loss_dfl_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=_base_.num_classes,
            use_ciou=True,
            topk=_base_.tal_topk,
            alpha=_base_.tal_alpha,
            beta=_base_.tal_beta,
            eps=1e-9)),
    test_cfg=_base_.model_test_cfg)
