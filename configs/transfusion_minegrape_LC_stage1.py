# ====================== MineGrape - LC Stage-1 (freeze LiDAR) ======================
dataset_type = 'NuScenesDataset'
data_root = 'data/test'
class_names = ('grape','leaf','stem','vine')
input_modality = dict(use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False)

# 单相机（默认 CAM_FRONT）
num_views = 1
img_scale = (800, 448)
img_norm_cfg = dict(mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True)

point_cloud_range = [-10.0, -10.0, -3.0, 10.0, 10.0, 3.0]
voxel_size = [0.05, 0.05, 0.10]
out_size_factor = 8

# ---------------------- Pipelines ----------------------
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),  # 单视角也复用该算子
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0,1,2,3]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.3925,0.3925], scale_ratio_range=[0.95,1.05], translation_std=[0,0,0]),
    dict(type='RandomFlip3D', sync_2d=True, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.0),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    # 你代码里有自定义的 MyResize/MyNormalize/MyPad（与 mini 配置一致）：
    dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
    dict(type='MyNormalize', **img_norm_cfg),
    dict(type='MyPad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d','gt_vis_labels','gt_cause_labels'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0,1,2,3]),
    dict(type='MultiScaleFlipAug3D', img_scale=img_scale, pts_scale_ratio=1, flip=False,
         transforms=[
            dict(type='GlobalRotScaleTrans', rot_range=[0,0], scale_ratio_range=[1.0,1.0], translation_std=[0,0,0]),
            dict(type='RandomFlip3D'),
            dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
            dict(type='MyNormalize', **img_norm_cfg),
            dict(type='MyPad', size_divisor=32),
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['points','img'])
         ])
]

# ---------------------- Dataset ----------------------
data = dict(
    samples_per_gpu=2, workers_per_gpu=6,
    train=dict(
        type='CBGSDataset',  # 与 mini 配置一致，按类均衡
        dataset=dict(
            type=dataset_type, data_root=data_root, num_views=num_views,
            ann_file=data_root + '/custom_infos_train.pkl',
            load_interval=1, pipeline=train_pipeline, classes=class_names,
            modality=input_modality, test_mode=False, box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type, data_root=data_root, num_views=num_views,
        ann_file=data_root + '/custom_infos_val.pkl',
        load_interval=1, pipeline=test_pipeline, classes=class_names,
        modality=input_modality, test_mode=True, box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type, data_root=data_root, num_views=num_views,
        ann_file=data_root + '/custom_infos_val.pkl',
        load_interval=1, pipeline=test_pipeline, classes=class_names,
        modality=input_modality, test_mode=True, box_type_3d='LiDAR')
)

# ---------------------- Model ----------------------
model = dict(
    type='TransFusionDetector',
    freeze_img=False,  # 训练相机分支
    # 图像 backbone: ResNet50，使用 ImageNet 预训练
    img_backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(0,1,2,3),
        frozen_stages=2, norm_cfg=dict(type='BN', requires_grad=True), norm_eval=True, style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    img_neck=dict(type='FPN', in_channels=[256,512,1024,2048], out_channels=256, num_outs=5),

    # LiDAR 分支保持与 L-only 一致（但**Stage-1 冻结**，见 optimizer.paramwise_cfg）
    pts_voxel_layer=dict(max_num_points=10, voxel_size=voxel_size, max_voxels=(120000,160000),
                         point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder', in_channels=4,
        sparse_shape=[41, 400, 400],  # 近似：Z40, XY各400
        output_channels=128, order=('conv','norm','act'),
        encoder_channels=((16,16,32),(32,32,64),(64,64,128),(128,128)),
        encoder_paddings=((0,0,1),(0,0,1),(0,0,[0,1,1]),(0,0)), block_type='basicblock'),
    pts_backbone=dict(type='SECOND', in_channels=256, out_channels=[128,256], layer_nums=[5,5], layer_strides=[1,2],
                      norm_cfg=dict(type='BN', eps=1e-3, momentum=1e-2), conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(type='SECONDFPN', in_channels=[128,256], out_channels=[256,256], upsample_strides=[1,2],
                  norm_cfg=dict(type='BN', eps=1e-3, momentum=1e-2), upsample_cfg=dict(type='deconv', bias=False),
                  use_conv_for_no_stride=True),

    pts_bbox_head=dict(
        type='TransFusionHead',
        fuse_img=True, num_views=num_views, in_channels_img=256, out_size_factor_img=4,
        num_proposals=200, auxiliary=True, in_channels=256*2, hidden_channel=128,
        num_classes=len(class_names), num_decoder_layers=1, num_heads=8,
        learnable_query_pos=False, initialize_by_heatmap=True, nms_kernel_size=3,
        ffn_channel=256, dropout=0.1, bn_momentum=0.1, activation='relu',
        common_heads=dict(center=(2,2), height=(1,2), dim=(3,2), rot=(2,2), vel=(2,2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder', pc_range=point_cloud_range[:2], voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor, post_center_range=[-61.2,-61.2,-10.0,61.2,61.2,10.0],
            score_threshold=0.0, code_size=10),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        with_aux_attr=True, num_vis=4, num_cause=6,
        loss_vis=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
        loss_cause=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
    ),
    train_cfg=dict(pts=dict(
        dataset='nuScenes',
        assigner=dict(type='HungarianAssigner3D',
            iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
            cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
            reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
            iou_cost=dict(type='IoU3DCost', weight=0.25)),
        pos_weight=-1, gaussian_overlap=0.1, min_radius=2,
        grid_size=[400,400,40], voxel_size=voxel_size, out_size_factor=out_size_factor,
        code_weights=[1.0]*8+[0.2,0.2], point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(dataset='nuScenes', grid_size=[400,400,40], out_size_factor=out_size_factor,
                           pc_range=point_cloud_range[0:2], voxel_size=voxel_size[:2], nms_type=None))
)

# ---------------------- Train (冻结 LiDAR) ----------------------
# 从 LiDAR-only 检查点加载，冻结 pts_*，仅训练图像分支 + 融合 + 头部
optimizer = dict(
    type='AdamW', lr=1e-4, weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={
        'pts_voxel_encoder': dict(lr_mult=0.0, decay_mult=0.0),
        'pts_middle_encoder': dict(lr_mult=0.0, decay_mult=0.0),
        'pts_backbone': dict(lr_mult=0.0, decay_mult=0.0),
        'pts_neck': dict(lr_mult=0.0, decay_mult=0.0),
        'img_backbone': dict(lr_mult=0.5),  # 预训练 R50，学习率略小更稳
    })
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='CosineAnnealing', min_lr=1e-6,
                 warmup='linear', warmup_iters=1000, warmup_ratio=0.1)
total_epochs = 10  # 暖起阶段不必太久
evaluation = dict(interval=1, pipeline=test_pipeline)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'),
                                      dict(type='TensorboardLoggerHook')])
checkpoint_config = dict(interval=1)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = '/path/to/your_lidar_only_checkpoint.pth'  # ← 放 LiDAR-only 的 ckpt
resume_from = None
find_unused_parameters = True
# 只训练
workflow = [('train', 1)]

# （如果你想每个 epoch 后做一次验证）
# workflow = [('train', 1), ('val', 1)]
