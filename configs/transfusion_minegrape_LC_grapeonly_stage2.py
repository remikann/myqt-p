# ====================== MineGrape ======================
# 数据与类别
dataset_type = 'NuScenesGrapeDataset'
data_root = 'data/test'  # 实际路径
class_names = ('grape',)

# 点云/体素设置（近景作物，先小范围，后续可调）
point_cloud_range = [-2.0, -2.0, -1.5, 2.0, 2.0, 1.5]
voxel_size = [0.0125, 0.0125, 0.10]
out_size_factor = 8
input_modality = dict(use_lidar=True, use_camera=True, use_radar=False, use_map=False, use_external=False)
img_norm_cfg = dict(mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True)
# 可选：先不开 DB 采样，链路稳定后再加
db_sampler = dict(
    data_root=data_root,
    info_path=data_root + '/custom_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1],
                 filter_by_min_points=dict(grape=1, leaf=1, stem=1, vine=1)),
    classes=list(class_names),
    sample_groups=dict(grape=20, leaf=20, stem=20, vine=20),
    points_loader=dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0,1,2,3]),
)
# train_pipeline.insert(3, dict(type='ObjectSample', db_sampler=db_sampler))


# ---------------------- Pipelines ----------------------
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),  # 虽然只有 CAM_FRONT，也可以用这个
    dict(type='MyResize', img_scale=(800,448), keep_ratio=True),
    dict(type='MyNormalize', **img_norm_cfg),
    dict(type='MyPad', size_divisor=32),

    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0,1,2,3]),  # 只用 XYZI
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    
     # ★ 把 ann_info 里的属性塞进 results（此时长度==当前 GT 数）
    dict(type='FetchAttrFromAnnInfo'),

    dict(type='RecordNumGT'),                             # ★ 新增
    # dict(type='ObjectSample', db_sampler=db_sampler),     # ★ DB 采样（只在 LiDAR-only 配置加）
    # dict(type='PadAttrAfterSample', ignore_index=-1),     # ★ 新增（给 DB 样本补属性 = -1）
    
    dict(type='GlobalRotScaleTrans', rot_range=[-0.7854, 0.7854], scale_ratio_range=[0.95, 1.05], translation_std=[0,0,0]),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='PackAttrForCollate'), 
    dict(
        type='Collect3D',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d',
                'gt_vis_labels', 'gt_cause_labels']
    )
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles'),  # 虽然只有 CAM_FRONT，也可以用这个
    dict(type='MyResize', img_scale=(800,448), keep_ratio=True),
    dict(type='MyNormalize', **img_norm_cfg),
    dict(type='MyPad', size_divisor=32),

    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0,1,2,3]),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['points','img'])
]

# ---------------------- Dataset & DB ----------------------
data = dict(
    samples_per_gpu=1, workers_per_gpu=4,
    train=dict(
        type=dataset_type, data_root=data_root,
        ann_file=data_root + '/custom_infos_train.pkl',  # 如用自定义名，改这里
        pipeline=train_pipeline, classes=class_names, modality=input_modality,
        use_valid_flag=True, box_type_3d='LiDAR'
    ),
    val=dict(
        type=dataset_type, data_root=data_root,
        ann_file=data_root + '/custom_infos_val.pkl',
        pipeline=test_pipeline, classes=class_names, modality=input_modality,
        test_mode=True, box_type_3d='LiDAR'
    ),
    test=dict(
        type=dataset_type, data_root=data_root,
        ann_file=data_root + '/custom_infos_val.pkl',
        pipeline=test_pipeline, classes=class_names, modality=input_modality,
        test_mode=True, box_type_3d='LiDAR'
    )
)


# ---------------------- Model (4 通道输入) ----------------------
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
    pts_voxel_layer=dict(
        max_num_points=10, voxel_size=voxel_size, max_voxels=(60000, 80000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder', in_channels=4,
        sparse_shape=[
            int((point_cloud_range[5]-point_cloud_range[2])/voxel_size[2])+1,
            int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1]),
            int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0])],
        output_channels=128, order=('conv','norm','act'),
        encoder_channels=((16,16,32),(32,32,64),(64,64,128),(128,128)),
        encoder_paddings=((0,0,1),(0,0,1),(0,0,[0,1,1]),(0,0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND', in_channels=128, out_channels=[128,256], layer_nums=[5,5], layer_strides=[1,2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=1e-2), conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN', in_channels=[128,256], out_channels=[256,256], upsample_strides=[1,2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=1e-2), upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='TransFusionHead',
        fuse_img=True,
        num_views=1, in_channels_img=256, out_size_factor_img=4,
        num_proposals=400, auxiliary=True, in_channels=256*2, hidden_channel=128,
        num_classes=len(class_names), num_decoder_layers=1, num_heads=8,
        learnable_query_pos=False, initialize_by_heatmap=True, 
        # learnable_query_pos=True, initialize_by_heatmap=False, 
        nms_kernel_size=3,
        ffn_channel=256, dropout=0.1, bn_momentum=0.1, activation='relu',
        common_heads=dict(center=(2,2), height=(1,2), dim=(3,2), rot=(2,2), vel=(2,2)),
        bbox_coder=dict(type='TransFusionBBoxCoder',
                        pc_range=point_cloud_range[:2], voxel_size=voxel_size[:2], out_size_factor=out_size_factor,
                        post_center_range=[-2.5, -2.5, -2, 2.5, 2.5, 2], score_threshold=0.0, code_size=10),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        # 如果你已实现 visibility/cause 两路损失，解开下面：
        with_aux_attr=True, num_vis=4, num_cause=6,
        loss_vis=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
        loss_cause=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
    ),
    train_cfg=dict(pts=dict(
        # dataset='nuScenes',
        dataset='nuScenes',
        assigner=dict(type='HungarianAssigner3D',
                      iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                      cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                      reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                      iou_cost=dict(type='IoU3DCost', weight=0.25)),
        pos_weight=-1, gaussian_overlap=0.3, min_radius=2,
        grid_size=[
            int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0]),
            int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1]),
            int((point_cloud_range[5]-point_cloud_range[2])/voxel_size[2])],
        voxel_size=voxel_size, out_size_factor=out_size_factor,
        code_weights=[1.0]*8 + [0.2,0.2], point_cloud_range=point_cloud_range)),
    test_cfg=dict(pts=dict(
        dataset='nuScenes',
        grid_size=[
            int((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0]),
            int((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1]),
            int((point_cloud_range[5]-point_cloud_range[2])/voxel_size[2])],
        out_size_factor=out_size_factor, pc_range=point_cloud_range[0:2], voxel_size=voxel_size[:2], nms_type=None))
)

# ---------------------- Train ----------------------
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={
                'img_backbone': dict(lr_mult=1),
                })
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='CosineAnnealing', min_lr=1e-6,
                 warmup='linear', warmup_iters=500, warmup_ratio=0.1)
total_epochs = 50
# evaluation = dict(interval=1, pipeline=test_pipeline)
evaluation = dict(interval=1)
log_config = dict(interval=2, hooks=[dict(type='TextLoggerHook'),
                                      dict(type='TensorboardLoggerHook')])
checkpoint_config = dict(interval=10)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = "work_dirs/minegrape_go_LC1/epoch_20.pth"
resume_from = None
find_unused_parameters = True
# 只训练
workflow = [('train', 1)]

# # （如果你想每个 epoch 后做一次验证）
# workflow = [('train', 1), ('val', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=50)