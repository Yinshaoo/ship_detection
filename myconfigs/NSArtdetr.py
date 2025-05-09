auto_scale_lr = dict(base_batch_size=16)
backend_args = None
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
data_root = 'data/coco/'
dataset_type = 'ShipDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook',max_keep_ckpts=3),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_size = (
    640,
    640,
)
find_unused_parameters = True
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 72
model = dict(
    as_two_stage=True,
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint=
            'https://github.com/nijkah/storage/releases/download/v0.0.1/resnet50vd_ssld_v2_pretrained.pth',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNetV1d'),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.75,
            gamma=2.0,
            loss_weight=1.0,
            type='VarifocalLoss',
            use_rtdetr=True,
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=1,
        type='RTDETRHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            0,
            0,
            0,
        ],
        pad_size_divisor=32,
        std=[
            255,
            255,
            255,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        eval_idx=-1,
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=3),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=None,
    eval_size=(
        640,
        640,
    ),
    neck=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                act_cfg=dict(type='GELU'),
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0),
            self_attn_cfg=dict(
                     embed_dims=256, num_heads=8,topk=4,
                     block_size=7, swin_size=7, qkv_bias=True,
                     qk_scale=None, drop_rate=0.2, attn_drop_rate=0.2,)),
        num_encoder_layers=1,
        projector=dict(
            act_cfg=None,
            in_channels=[
                256,
                256,
                256,
            ],
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            num_outs=3,
            out_channels=256,
            type='ChannelMapper'),
        type='NSAHybridEncoder'),
    num_queries=300,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='RTDETR',
    with_box_refine=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
#     dict(
#         type='CosineAnnealingLR',
#         eta_min=0.0001 * 0.001,
#         begin=1,
#         end=max_epochs,
#         T_max=max_epochs,
#         by_epoch=True
#     )
# ]
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=2000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[100],
        gamma=1.0)
]
pretrained = 'https://github.com/nijkah/storage/releases/download/v0.0.1/resnet50vd_ssld_v2_pretrained.pth'
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                interpolation='bicubic',
                keep_ratio=False,
                scale=(
                    640,
                    640,
                ),
                type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='ShipDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        interpolation='bicubic',
        keep_ratio=False,
        scale=(
            640,
            640,
        ),
        type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=6)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=16,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        backend_args=None,
        data_prefix=dict(img='train2017/'),
        data_root='data/coco/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=False,
                scales=[
                    (
                        480,
                        480,
                    ),
                    (
                        512,
                        512,
                    ),
                    (
                        544,
                        544,
                    ),
                    (
                        576,
                        576,
                    ),
                    (
                        608,
                        608,
                    ),
                    (
                        640,
                        640,
                    ),
                    (
                        640,
                        640,
                    ),
                    (
                        640,
                        640,
                    ),
                    (
                        672,
                        672,
                    ),
                    (
                        704,
                        704,
                    ),
                    (
                        736,
                        736,
                    ),
                    (
                        768,
                        768,
                    ),
                    (
                        800,
                        800,
                    ),
                ],
                type='RandomChoiceResize'),
            dict(type='PhotoMetricDistortion'),
            dict(
                mean=[
                    123.675,
                    116.28,
                    103.53,
                ],
                ratio_range=(
                    1,
                    2,
                ),
                to_rgb=True,
                type='Expand'),
            dict(crop_size=(
                640,
                640,
            ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(type='PackDetInputs'),
        ],
        type='ShipDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=False,
        scales=[
            (
                480,
                480,
            ),
            (
                512,
                512,
            ),
            (
                544,
                544,
            ),
            (
                576,
                576,
            ),
            (
                608,
                608,
            ),
            (
                640,
                640,
            ),
            (
                640,
                640,
            ),
            (
                640,
                640,
            ),
            (
                672,
                672,
            ),
            (
                704,
                704,
            ),
            (
                736,
                736,
            ),
            (
                768,
                768,
            ),
            (
                800,
                800,
            ),
        ],
        type='RandomChoiceResize'),
    dict(type='PhotoMetricDistortion'),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        ratio_range=(
            1,
            2,
        ),
        to_rgb=True,
        type='Expand'),
    dict(crop_size=(
        640,
        640,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                interpolation='bicubic',
                keep_ratio=False,
                scale=(
                    640,
                    640,
                ),
                type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='ShipDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend', init_kwargs=dict(project='rtdetr', group='rtdetr', name='rtdetr-defult'))
                ]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

work_dir = './work_dirs/RTDETR_NSA'
