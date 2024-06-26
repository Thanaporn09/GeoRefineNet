_base_ = [
    './mmpose/configs/_base_/default_runtime.py',
    './mmpose/configs/_base_/datasets/Head_dataset.py'
]
checkpoint_config = dict(interval=100)
evaluation = dict(interval=1, metric=['MRE_h','MRE_std_h','SDR_2h','SDR_2.5h','SDR_3h','SDR_4h'], save_best='MRE_h')

optimizer = dict(
    type='AdamW',
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    )

optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1e-5,
    warmup_by_epoch=True)

total_epochs = 50
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=13,
    dataset_joints=13,
    dataset_channel=[
        list(range(13)),
    ],
    inference_channel=list(range(13)))
norm_cfg = dict(type='SyncBN', requires_grad=True)

# model settings
model = dict(
    type='TopDown_2Head_Neck_3stage',
    backbone=dict(
        type='HTC_v1_Sigmoid',
        num_stages=4,
        num_layers=[1, 1, 1, 1],
        patch_sizes=[3, 3, 3, 3],
        strides=[2, 2, 2, 2],
        paddings=[1, 1, 1, 1],
        num_heads=[1, 2, 5, 8],
        chratio=[8, 4, 2, 1],
        out_indices=(0,1,2,3)),
    neck=dict(
        type='Deconv_FPN_Neck',
        in_channels = [64,128,320,512],
        out_channels = 256,
        num_deconv_layers=2,
        num_deconv_filters=(320, 128),
        num_deconv_kernels=(4, 4),
    ),
    distance_head = dict(
        type='TopdownSimple_Distance_DeConvHead',
        in_channels=256,
        feat_channels=256,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint_reg=dict(type='JointsMSELoss', use_target_weight=True,loss_weight=2000)
    ),
    keypoint_head=dict(
        type='TopdownSimple_Heatmap_ConvHead',
        in_channels=256,
        feat_channels=256,
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True,loss_weight=1000)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11,
        use_udp=False))

data_cfg = dict(
    image_size=[608,480],
    heatmap_size=[152,120], 
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file=''
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='attention_heatmap_loading'),
    dict(type='TopDownRandomFlip_Heatmap', flip_prob=0.5),
    dict(type='PhotometricDistortion'),
    dict(type='TopDownAffine_Heatmap'),
    dict(
        type='NormalizeTensor', 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        to_rgb=True),
    dict(
        type='NormalizeTensor_Heatmap', 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        to_rgb=True),
    dict(type='ToTensor'),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(type='Topdown_Heatmap_Target_Size',sigma=2,size=2),
    dict(type='ToTensor_Heatmap'),
    dict(
        type='Collect',
        keys=['img','target','target_weight', 'attention_heatmap','target_reg','target_weight_reg'],
        meta_keys=[
            'image_file','center', 'scale', 'joints_3d', 'joints_3d_visible',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='attention_heatmap_loading'),
    dict(type='TopDownAffine_Heatmap'),
    dict(
        type='NormalizeTensor_Heatmap', 
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        to_rgb=True),
    dict(type='ToTensor'),
    dict(type='ToTensor_Heatmap'),
    dict(
        type='Collect',
        keys=['img', 'attention_heatmap'],
        meta_keys=[
            'image_file','center', 'scale', 'rotation', 'bbox_score','joints_3d',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline
datatype='TopDownHeadDataset_Heatmap'
data_root = './Head_Dataset/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2),
    train=dict(
        type=datatype,
        attention_heatmap_prefix=f'{data_root}/2D_Heatmaps/train',
        ann_file=f'{data_root}/JSON/Head_dataset_train.json',
        img_prefix=f'{data_root}/2D_img/train',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type=datatype,
        attention_heatmap_prefix=f'{data_root}/2D_Heatmaps/test',
        ann_file=f'{data_root}/JSON/Head_dataset_test.json',
        img_prefix=f'{data_root}/2D_img/test',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type=datatype,
        attention_heatmap_prefix=f'{data_root}/2D_Heatmaps/test',
        ann_file=f'{data_root}/JSON/Head_dataset_test.json',
        img_prefix=f'{data_root}/2D_img/test',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)

work_dir = ''