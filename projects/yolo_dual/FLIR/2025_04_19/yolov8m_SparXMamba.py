_base_ = '/data/nl/Mamba2Spectral/projects/yolo_dual/datasets/FILR.py'


model = dict(
    type='YOLODualNeckDetector',
    data_preprocessor=dict(
        type='DualInputDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    pre_handle = dict(
        type = 'Identity',
    ),
    backbone=dict(
        type='GeneralDualBackbone',
        _delete_=True,
        stages=dict(
            type='sparXMamba',
            settings=dict(
                in_channels=[48, 96, 192, 328],
                out_channels=[96, 192, 328, 544],
                patch_embed = [True,False,False,False],
                depth=[2,2,17,2],
                sr_ratio=[8, 4, 2, 1],
                is_first_stage=[True, False, False, False],
                max_dense_depth = [100,100,3,100],
                dense_step = [1,1,3,1],
                dense_start = [100, 1, 0, 0],
                widen_factor=_base_.widen_factor,
                deepen_factor=_base_.deepen_factor,
                
            )
        ),
        input_channels=3,
        stem_out_channels=48,
        norm_cfg=_base_.norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        deepen_factor=_base_.deepen_factor,
        widen_factor=_base_.widen_factor,
        out_indices= (2, 3, 4),
        fusion_indices=(2,3, 4,),
        fusion_block=dict(
                type = 'MM_SS2D',
                in_channels = [96, 192, 328],
                size = [(128 , 160),(64 , 80), (32 , 40)],
                mamba_opreations = [1 , 1 , 1],
                bi = [False , False , True]
            ),
        fusion_module = dict(type = 'CommonConv' , in_channels = [96, 192, 328]),
        head_input_module = dict(type = 'CommonConv' , in_channels = [192, 328, 544]),
    ),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=_base_.deepen_factor,
        widen_factor=_base_.widen_factor,
        in_channels=[192, 328, 544],
        out_channels=[192, 328, 544],
        num_csp_blocks=3,
        norm_cfg=_base_.norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=_base_.num_classes,
            in_channels=[192, 328, 544],
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
