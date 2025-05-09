_base_ = '/data/nl/Mamba2Spectral/projects/yolo_dual/datasets/FILR.py'


model = dict(
    type='YOLODualNeckDetector',
    data_preprocessor=dict(
        type='DualInputDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='GeneralDualBackbone',
        _delete_=True,
        fusion_flag= True,
        stages=dict(
            type='yoloV8CSPDarknet',
            settings=dict(
                in_channels=[64,128,256,512],
                out_channels=[128,256,512,1024],
                num_blocks=[3,6,6,3],
                add_identity=[True, True, True, True],
                use_spp=[False, False, False, True],
                widen_factor=_base_.widen_factor,
                deepen_factor=_base_.deepen_factor,
                norm_cfg=_base_.norm_cfg,
                act_cfg=dict(type='SiLU', inplace=True)
            )
        ),
        input_channels=3,
        norm_cfg=_base_.norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
        deepen_factor=_base_.deepen_factor,
        widen_factor=_base_.widen_factor,
        out_indices= (2, 3, 4),
        fusion_indices=(2,3, 4,),
        stem_block=dict(
            type = 'ConvStem',
            widen_factor = _base_.widen_factor,
            in_channels = 3,
            out_channels = 64,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            norm_cfg = _base_.norm_cfg,
            act_cfg = dict(type='SiLU', inplace=True)
        ),
        fusion_block=dict(
                type = 'DMFF',
                in_channels = [128 , 256 , 512],
                vert_anchor = [16 , 10 , 8], 
                horz_anchor = [16 , 10 , 8],
                loops_num = 1,
                fusion = True,
                mod_weight = True
        ),
        fusion_module = dict(type = 'BaseFusion',in_channels = [128,256,512],),
        head_input_module = dict(type = 'BaseFusion',in_channels = [256,512,1024],),
    ),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=_base_.deepen_factor,
        widen_factor=_base_.widen_factor,
        in_channels=[256, 512, _base_.last_stage_out_channels],
        out_channels=[256, 512, _base_.last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=_base_.norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=_base_.num_classes,
            in_channels=[256, 512, _base_.last_stage_out_channels],
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
