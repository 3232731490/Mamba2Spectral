_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
affine_scale = 0.5
after_pre_transformer = [
    dict(
        border=(
            -320,
            -256,
        ),
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='YOLOv5RandomAffine'),
]
albu_train_transforms = [
    dict(p=0.01, type='Blur'),
    dict(p=0.01, type='MedianBlur'),
    dict(p=0.01, type='ToGray'),
    dict(p=0.01, type='CLAHE'),
]
backend_args = None
base_lr = 0.01
batch_shapes_cfg = None
before_last_transformer = [
    dict(type='YOLOv5HSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
]
class_name = (
    'car',
    'person',
    'bicycle',
)
classes = (
    'car',
    'person',
    'bicycle',
)
close_mosaic_epochs = 10
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=190,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(backend_args=None, type='LoadImageFromFile2'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(
                transforms=[
                    dict(scale=(
                        640,
                        512,
                    ), type='YOLOv5KeepRatioResize'),
                    dict(
                        border_val=(
                            114,
                            114,
                            114,
                        ),
                        max_aspect_ratio=100,
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(
                            0.5,
                            1.5,
                        ),
                        type='YOLOv5RandomAffine'),
                ],
                type='Image2Broadcaster'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='Dual_Albu'),
            dict(
                transforms=[
                    dict(type='YOLOv5HSVRandomAug'),
                    dict(prob=0.5, type='mmdet.RandomFlip'),
                ],
                type='Image2Broadcaster'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'img_path2',
                    'ori_shape2',
                    'img_shape2',
                    'scale_factor',
                ),
                type='DoublePackDetInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
data_root = '/data/nl/mmdet2spectral/data/FLIR/'
dataset_type = 'DualStreamCocoDataset'
deepen_factor = 0.67
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP_50',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=200,
        scheduler_type='linear',
        type='YOLOv5ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    640,
    512,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
last_stage_out_channels = 1024
last_transform = [
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='Dual_Albu'),
    dict(
        transforms=[
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
        ],
        type='Image2Broadcaster'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'img_path2',
            'ori_shape2',
            'img_shape2',
            'scale_factor',
        ),
        type='DoublePackDetInputs'),
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 7.5
loss_cls_weight = 0.5
loss_dfl_weight = 0.375
lr_factor = 0.01
max_aspect_ratio = 100
max_epochs = 200
max_keep_ckpts = 3
metainfo = dict(
    classes=(
        'car',
        'person',
        'bicycle',
    ),
    palette=[
        (
            20,
            220,
            60,
        ),
        (
            119,
            11,
            32,
        ),
        (
            0,
            0,
            142,
        ),
    ])
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.67,
        fusion_block=dict(in_channels=[
            128,
            256,
            512,
        ], type='BaseFusion'),
        fusion_flag=True,
        fusion_indices=(
            2,
            3,
            4,
        ),
        fusion_module=dict(in_channels=[
            128,
            256,
            512,
        ], type='BaseFusion'),
        head_input_module=dict(
            in_channels=[
                256,
                512,
                1024,
            ], type='BaseFusion'),
        input_channels=3,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        out_indices=(
            2,
            3,
            4,
        ),
        stages=dict(
            settings=dict(
                act_cfg=dict(inplace=True, type='SiLU'),
                add_identity=[
                    True,
                    True,
                    True,
                    True,
                ],
                deepen_factor=0.67,
                in_channels=[
                    64,
                    128,
                    256,
                    512,
                ],
                norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
                num_blocks=[
                    3,
                    6,
                    6,
                    3,
                ],
                out_channels=[
                    128,
                    256,
                    512,
                    1024,
                ],
                use_spp=[
                    False,
                    False,
                    False,
                    True,
                ],
                widen_factor=0.75),
            type='yoloV8CSPDarknet'),
        stem_block=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            in_channels=3,
            kernel_size=3,
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            out_channels=64,
            padding=1,
            stride=2,
            type='ConvStem',
            widen_factor=0.75),
        type='GeneralDualBackbone',
        widen_factor=0.75),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=[
                256,
                512,
                1024,
            ],
            norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
            num_classes=3,
            reg_max=16,
            type='YOLOv8HeadModule',
            widen_factor=0.75),
        loss_bbox=dict(
            bbox_format='xyxy',
            iou_mode='ciou',
            loss_weight=7.5,
            reduction='sum',
            return_iou=False,
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=0.5,
            reduction='none',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True),
        loss_dfl=dict(
            loss_weight=0.375,
            reduction='mean',
            type='mmdet.DistributionFocalLoss'),
        prior_generator=dict(
            offset=0.5, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='YOLOv8Head'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DualInputDetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=0.67,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_csp_blocks=3,
        out_channels=[
            256,
            512,
            1024,
        ],
        type='YOLOv8PAFPN',
        widen_factor=0.75),
    test_cfg=dict(
        max_per_img=300,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        assigner=dict(
            alpha=0.5,
            beta=6.0,
            eps=1e-09,
            num_classes=3,
            topk=10,
            type='BatchTaskAlignedAssigner',
            use_ciou=True)),
    type='YOLODualNeckDetector')
model_test_cfg = dict(
    max_per_img=300,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=30000,
    score_thr=0.001)
norm_cfg = dict(eps=0.001, momentum=0.03, type='BN')
num_classes = 3
num_det_layers = 3
optim_wrapper = dict(
    clip_grad=dict(max_norm=10.0),
    constructor='YOLOv5OptimizerConstructor',
    optimizer=dict(
        batch_size_per_gpu=8,
        lr=0.01,
        momentum=0.937,
        nesterov=True,
        type='SGD',
        weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = None
persistent_workers = True
pre_transform = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(backend_args=None, type='LoadImageFromFile2'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
]
resume = True
save_epoch_intervals = 1
strides = [
    8,
    16,
    32,
]
tal_alpha = 0.5
tal_beta = 6.0
tal_topk = 10
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val_class3.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='val/enhance/rgb'),
        data_root='/data/nl/mmdet2spectral/data/FLIR/',
        metainfo=dict(
            classes=(
                'car',
                'person',
                'bicycle',
            ),
            palette=[
                (
                    20,
                    220,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(backend_args=None, type='LoadImageFromFile2'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                transforms=[
                    dict(scale=(
                        640,
                        512,
                    ), type='YOLOv5KeepRatioResize'),
                ],
                type='Image2Broadcaster'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'img_path2',
                    'ori_shape2',
                    'img_shape2',
                    'scale_factor',
                ),
                type='DoublePackDetInputs'),
        ],
        test_mode=True,
        type='DualStreamCocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/data/nl/mmdet2spectral/data/FLIR/val_class3.json',
    classwise=True,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(backend_args=None, type='LoadImageFromFile2'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        transforms=[
            dict(scale=(
                640,
                512,
            ), type='YOLOv5KeepRatioResize'),
        ],
        type='Image2Broadcaster'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'img_path2',
            'ori_shape2',
            'img_shape2',
            'scale_factor',
        ),
        type='DoublePackDetInputs'),
]
test_transformer = [
    dict(scale=(
        640,
        512,
    ), type='YOLOv5KeepRatioResize'),
]
train_ann_file = 'train_class3.json'
train_batch_size_per_gpu = 8
train_cfg = dict(
    dynamic_intervals=[
        (
            190,
            1,
        ),
    ],
    max_epochs=200,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_data_prefix = 'train/enhance/rgb'
train_dataloader = dict(
    batch_size=8,
    collate_fn=dict(type='dual_yolo_collate'),
    dataset=dict(
        ann_file='train_class3.json',
        data_prefix=dict(img='train/enhance/rgb'),
        data_root='/data/nl/mmdet2spectral/data/FLIR/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'car',
                'person',
                'bicycle',
            ),
            palette=[
                (
                    20,
                    220,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(backend_args=None, type='LoadImageFromFile2'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(
                img_scale=(
                    640,
                    512,
                ),
                pad_val=114.0,
                pre_transform=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(backend_args=None, type='LoadImageFromFile2'),
                    dict(
                        type='LoadAnnotations',
                        with_bbox=True,
                        with_mask=False),
                ],
                type='Dual_Mosaic',
                use_cached=False),
            dict(
                transforms=[
                    dict(
                        border=(
                            -320,
                            -256,
                        ),
                        border_val=(
                            114,
                            114,
                            114,
                        ),
                        max_aspect_ratio=100,
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(
                            0.5,
                            1.5,
                        ),
                        type='YOLOv5RandomAffine'),
                ],
                type='Image2Broadcaster'),
            dict(
                bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ],
                type='Dual_Albu'),
            dict(
                transforms=[
                    dict(type='YOLOv5HSVRandomAug'),
                    dict(prob=0.5, type='mmdet.RandomFlip'),
                ],
                type='Image2Broadcaster'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'img_path2',
                    'ori_shape2',
                    'img_shape2',
                    'scale_factor',
                ),
                type='DoublePackDetInputs'),
        ],
        type='DualStreamCocoDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 8
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(backend_args=None, type='LoadImageFromFile2'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        img_scale=(
            640,
            512,
        ),
        pad_val=114.0,
        pre_transform=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(backend_args=None, type='LoadImageFromFile2'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
        ],
        type='Dual_Mosaic',
        use_cached=False),
    dict(
        transforms=[
            dict(
                border=(
                    -320,
                    -256,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='YOLOv5RandomAffine'),
        ],
        type='Image2Broadcaster'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='Dual_Albu'),
    dict(
        transforms=[
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
        ],
        type='Image2Broadcaster'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'img_path2',
            'ori_shape2',
            'img_shape2',
            'scale_factor',
        ),
        type='DoublePackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(backend_args=None, type='LoadImageFromFile2'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        transforms=[
            dict(scale=(
                640,
                512,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type='YOLOv5RandomAffine'),
        ],
        type='Image2Broadcaster'),
    dict(
        bbox_params=dict(
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
                'gt_ignore_flags',
            ],
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', img='image'),
        transforms=[
            dict(p=0.01, type='Blur'),
            dict(p=0.01, type='MedianBlur'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='CLAHE'),
        ],
        type='Dual_Albu'),
    dict(
        transforms=[
            dict(type='YOLOv5HSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
        ],
        type='Image2Broadcaster'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'img_path2',
            'ori_shape2',
            'img_shape2',
            'scale_factor',
        ),
        type='DoublePackDetInputs'),
]
train_stage2_transformer = [
    dict(scale=(
        640,
        512,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        border_val=(
            114,
            114,
            114,
        ),
        max_aspect_ratio=100,
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(
            0.5,
            1.5,
        ),
        type='YOLOv5RandomAffine'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_ann_file = 'val_class3.json'
val_batch_size_per_gpu = 1
val_cfg = dict(type='ValLoop')
val_data_prefix = 'val/enhance/rgb'
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val_class3.json',
        batch_shapes_cfg=None,
        data_prefix=dict(img='val/enhance/rgb'),
        data_root='/data/nl/mmdet2spectral/data/FLIR/',
        metainfo=dict(
            classes=(
                'car',
                'person',
                'bicycle',
            ),
            palette=[
                (
                    20,
                    220,
                    60,
                ),
                (
                    119,
                    11,
                    32,
                ),
                (
                    0,
                    0,
                    142,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(backend_args=None, type='LoadImageFromFile2'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                transforms=[
                    dict(scale=(
                        640,
                        512,
                    ), type='YOLOv5KeepRatioResize'),
                ],
                type='Image2Broadcaster'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'img_path2',
                    'ori_shape2',
                    'img_shape2',
                    'scale_factor',
                ),
                type='DoublePackDetInputs'),
        ],
        test_mode=True,
        type='DualStreamCocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/data/nl/mmdet2spectral/data/FLIR/val_class3.json',
    classwise=True,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_interval_stage2 = 1
val_num_workers = 2
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
weight_decay = 0.0005
widen_factor = 0.75
work_dir = 'work_dirs/FLIR/2025_05_15/yolov8_Backbone_enhance'
