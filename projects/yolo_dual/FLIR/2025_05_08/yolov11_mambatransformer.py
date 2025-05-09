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
            type='yolov11Backbone',
            settings=dict(
                in_channels=[64,256,512,512],
                out_channels=[256,512,512,1024],
                hiden_channels = [128,256,512,1024],   
                num_blocks=[2,2,2,2],
                c3k=[False, False, True, True],
                e = [0.25,0.25,0.50,0.50],
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
                type = 'MambaTransformerBlock',
                in_channels = [256,512,512],
                mamba_size = [(128,160), (64 , 80), (32 , 40)],
                mamba_operations = [1 , 1 , 1],
                mamba_bi = [False , False , False],
                transformer_vert_anchor = [16 , 10 , 8], 
                transformer_horz_anchor = [16 , 10 , 8],
                transformer_loops_num = 1,
                transformer_fusion = True,
                transformer_mod_weight = True
        ),
        fusion_module = dict(type = 'BaseFusion',in_channels = [256,512,512],),
        head_input_module = dict(type = 'BaseFusion',in_channels = [512,512,1024],),
    ),
    neck=dict(
        type='YOLOv11PAFPN',
        deepen_factor=_base_.deepen_factor,
        widen_factor=_base_.widen_factor,
        in_channels=[512, 512, _base_.last_stage_out_channels],
        out_channels=[512, 512, _base_.last_stage_out_channels],
        num_csp_blocks=2,
        c3k = [False, False, True],
        norm_cfg=_base_.norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=_base_.num_classes,
            in_channels=[512, 512, _base_.last_stage_out_channels],
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
