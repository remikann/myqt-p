_base_ = ['./transfusion_nusc_voxel_L.py']

data_root = '/wangjinhai1/zYu/QTNet/data/mine'

point_cloud_range = [-20.0, -20.0, -3.0, 20.0, 20.0, 3.0]
class_names = [
    'car','truck','construction_vehicle','bus','trailer','barrier',
    'motorcycle','bicycle','pedestrian','traffic_cone'
]
input_modality = dict(use_lidar=True, use_camera=False, use_radar=False, use_map=False)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0,1,2,3,4]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='GlobalRotScaleTrans', rot_range=[-0.78539816,0.78539816], scale_ratio_range=[0.9,1.1], translation_std=[0,0,0]),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points','gt_bboxes_3d','gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0,1,2,3,4]),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='Collect3D', keys=['points'])
]

dataset_type = 'NuScenesDataset'
data = dict(
    samples_per_gpu=1, workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_root=data_root,
        ann_file=data_root + '/mine_infos_train.pkl',
        pipeline=train_pipeline, classes=class_names,
        modality=input_modality, use_valid_flag=True),
    val=dict(
        type=dataset_type, data_root=data_root,
        ann_file=data_root + '/mine_infos_val.pkl',
        pipeline=test_pipeline, classes=class_names,
        modality=input_modality),
    test=dict(
        type=dataset_type, data_root=data_root,
        ann_file=data_root + '/mine_infos_val.pkl',
        pipeline=test_pipeline, classes=class_names,
        modality=input_modality)
)
# 追加/覆盖到 transfusion_mine_voxel_L.py 里（与 _base_ 合并时会生效）
model = dict(
    pts_voxel_layer=dict(
        # 单帧最多保留的体素数（train, test）
        max_voxels=(2000, 3000),
        # 每个体素最多保留点数（默认 5 就行）
        max_num_points=5
    )
)

evaluation = dict(interval=1, pipeline=test_pipeline)
load_from = None
