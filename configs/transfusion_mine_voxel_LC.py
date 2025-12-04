# 继承你原本的 LiDAR+CAM 配置
_base_ = ['./transfusion_nusc_voxel_LC.py']

# 路径改为你的数据根
data_root = '/wangjinhai1/zYu/QTNet/data/mine'

# 关键：把多视角改为单视角（只用 CAM_FRONT）
# 大多数 TransFusion 配置会在这些位置用到视角数，统一改为 1
num_views = 1

# 如果你的基础配置里有这些字段，就覆盖掉（不在则忽略，不影响运行）
model = dict(
    # 有些版本把 num_views 写在 head 里
    pts_bbox_head=dict(num_views=1),
    # 有些把相机数量写在视角变换模块里
    img_view_transformer=dict(num_cams=1)
)

# 单视角的图像加载与点云加载 pipeline
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_modality = dict(
    use_lidar=True, use_camera=True, use_radar=False, use_map=False
)

# 注意：这里把 LoadMultiViewImageFromFiles 的 num_views 改成 1
train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        num_views=1),                       # 只取 CAM_FRONT
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0,1,2,3,4]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    # 建议同样先不做 DB-Sampler（多模态初跑链路）
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=True,                        # 多模态建议同步翻转
        flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage',
         mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D',
         keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        num_views=1),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0,1,2,3,4]),
    dict(type='NormalizeMultiviewImage',
         mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['img', 'points'])
]

dataset_type = 'NuScenesDataset'
data = dict(
    samples_per_gpu=2,      # 多模态显存占用更大，batch调小点
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/mine_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        use_valid_flag=True),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/mine_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/mine_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality)
)

evaluation = dict(interval=1, pipeline=test_pipeline)

# 不加载 6 视角的预训练（结构不匹配）；先从头前向/训练
load_from = None
