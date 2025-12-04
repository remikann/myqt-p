import mmcv
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp
import torch
from mmdet.datasets import DATASETS
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .custom_3d import Custom3DDataset
def _match_by_center(pred_xy, gt_xy, dist_thr):
    """贪心匹配：按预测顺序，与最近 GT 匹配且距离<=阈值；返回 tp、fp 和匹配对索引。
    pred_xy: (P,2) torch
    gt_xy:   (G,2) torch
    """
    device = pred_xy.device
    gt_xy = gt_xy.to(device)

    P = pred_xy.shape[0]
    G = gt_xy.shape[0]

    if P == 0:
        return (
            torch.zeros(0, dtype=torch.float32, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            []
        )
    if G == 0:
        return (
            torch.zeros(P, dtype=torch.float32, device=device),
            torch.ones(P, dtype=torch.float32, device=device),
            []
        )

    used = torch.zeros(G, dtype=torch.bool, device=pred_xy.device)
    tp = torch.zeros(P, dtype=torch.float32, device=pred_xy.device)
    fp = torch.zeros(P, dtype=torch.float32, device=pred_xy.device)
    matches = []
    for i in range(P):
        d = torch.norm(pred_xy[i][None, :] - gt_xy, dim=1)  # (G,)
        minv, j = torch.min(d, dim=0)
        if (minv.item() <= dist_thr) and (not used[j]):
            tp[i] = 1.0
            used[j] = True
            matches.append((i, int(j.item())))
        else:
            fp[i] = 1.0
    return tp, fp, matches

def _voc_ap(rec, prec):
    """11-point 风格的 AP 近似；rec/prec 为 numpy 数组。"""
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)

def _axis_aligned_iou_3d_np(box1, box2):
    """用轴对齐包围盒计算 3D IoU（忽略朝向 yaw）

    box: [x, y, z, dx, dy, dz, (yaw...)]
    这里假设框的尺寸是以中心为原点的 dx, dy, dz
    """
    x1, y1, z1, dx1, dy1, dz1 = box1[:6]
    x2, y2, z2, dx2, dy2, dz2 = box2[:6]

    # 还原成 [min, max]
    x1_min, x1_max = x1 - dx1 * 0.5, x1 + dx1 * 0.5
    y1_min, y1_max = y1 - dy1 * 0.5, y1 + dy1 * 0.5
    z1_min, z1_max = z1 - dz1 * 0.5, z1 + dz1 * 0.5

    x2_min, x2_max = x2 - dx2 * 0.5, x2 + dx2 * 0.5
    y2_min, y2_max = y2 - dy2 * 0.5, y2 + dy2 * 0.5
    z2_min, z2_max = z2 - dz2 * 0.5, z2 + dz2 * 0.5

    inter_dx = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_dy = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_dz = max(0.0, min(z1_max, z2_max) - max(z1_min, z2_min))
    inter_vol = inter_dx * inter_dy * inter_dz

    vol1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    vol2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)

    if vol1 <= 0.0 or vol2 <= 0.0:
        return 0.0

    union = vol1 + vol2 - inter_vol
    if union <= 0.0:
        return 0.0

    return float(inter_vol / (union + 1e-7))

@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    CLASSES = ('grape','leaf','stem','vine')

    def __init__(self,
                 ann_file,
                 num_views=6,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.num_views = num_views
        assert self.num_views <= 6
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        # —— 兼容自定义类：若评测配置没有你的类，就给个无限范围，避免 KeyError 与误过滤
        try:
            cr = self.eval_detection_configs.class_range
        except Exception:
            cr = {}
            self.eval_detection_configs.class_range = cr

        for cname in self.CLASSES:
            cr.setdefault(cname, float('inf'))
            
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        cam_orders = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            # for cam_type, cam_info in info['cams'].items():
            for cam_type in cam_orders:
                if cam_type not in info['cams']:
                    continue
                cam_info = info['cams'][cam_type]
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        ################加入visibility和cause属性预测##################
        # 取原始字段（可能不存在）
        gt_vis = info.get('gt_vis_labels', None)
        gt_cause = info.get('gt_cause_labels', None)

        # 先按 mask 过滤
        if gt_vis is not None:
            gt_vis = gt_vis[mask]
        if gt_cause is not None:
            gt_cause = gt_cause[mask]

        # —— 关键：保证始终有这两个键，并与 gt 个数对齐 ——
        n = len(gt_labels_3d)
        if gt_vis is None:
            gt_vis = np.full((n,), -1, dtype=np.int64)         # -1 表示忽略监督
        if gt_cause is None:
            gt_cause = np.full((n,), -1, dtype=np.int64)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_vis_labels=gt_vis,           # 无论如何都带上
            gt_cause_labels=gt_cause,
        )
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not isinstance(results[0], dict):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir
    
    def evaluate(self,
                results,
                metric='bbox',
                logger=None,
                jsonfile_prefix=None,
                result_names=['pts_bbox'],
                show=False,
                out_dir=None):
        """自定义评测：支持4类 grapes 数据集；默认用中心距离匹配算 mAP，
        并在可用时计算 vis/cause 属性准确率。
        - results: 来自 simple_test 的列表，每个元素是 {'pts_bbox': {...}}。
        - 我们直接用 results 和 GT 做评测，不再调用 NuScenes 官方评测器。
        """
        from collections import defaultdict
        import torch
        all_matchious = []
        assert isinstance(results, list) and len(results) == len(self), \
            f'Expect len(results)==len(dataset), got {len(results)} vs {len(self)}'

        # 评测参数（可在 cfg.data.val/eval_params 里通过 dataset 初始化传入）
        eval_params = getattr(self, 'eval_params', {})
        # 中心距离阈值（米），用于 TP/FP 匹配；你也可以设不同类不同阈值
        dist_thr = float(eval_params.get('center_dist_thr', 0.05))  # 默认 5cm
        # AP 计算时的分阈值（给你常用两档，后续可以拓展更多）
        ap_thrs = eval_params.get('ap_thrs', [0.25, 0.5])  # 名义占位；中心匹配时我们只会算一档
        # 匹配模式：'center'（默认）、'bev_iou'、'iou_3d'
        match_mode = eval_params.get('match_mode', 'center')

        # 统计结构
        cls2scores = {c: [] for c in self.CLASSES}
        cls2tp = {c: [] for c in self.CLASSES}
        cls2fp = {c: [] for c in self.CLASSES}
        cls2gt = {c: 0 for c in self.CLASSES}

        # 属性统计（可选）
        do_attr = True
        attr_hit = {'vis': 0, 'cause': 0}
        attr_tot = {'vis': 0, 'cause': 0}
        # 属性统计（只在有匹配对、且 pred/gt 都带该属性时计数）
        # if do_attr and len(matches) > 0 and attrs_pred is not None:
        #     m_pred_mask = pred_mask.nonzero(as_tuple=False).squeeze(1)[sel_inds]
        #     m_gt_idx = gt_mask.nonzero(as_tuple=False).squeeze(1)

        #     # vis
        #     if ('vis' in attrs_pred) and (gt_vis is not None):
        #         for p_idx, g_local in matches:
        #             g_idx = m_gt_idx[g_local]
        #             # 只统计 grape/stem（假设其它类的 gt_vis 设为 -1）
        #             if int(gt_vis[g_idx]) < 0:
        #                 continue
        #             p_idx_global = m_pred_mask[p_idx]
        #             if int(attrs_pred['vis'][p_idx_global]) == int(gt_vis[g_idx]):
        #                 attr_hit['vis'] += 1
        #             attr_tot['vis'] += 1

        #     # cause
        #     if ('cause' in attrs_pred) and (gt_cause is not None):
        #         for p_idx, g_local in matches:
        #             g_idx = m_gt_idx[g_local]
        #             if int(gt_cause[g_idx]) < 0:
        #                 continue
        #             p_idx_global = m_pred_mask[p_idx]
        #             if int(attrs_pred['cause'][p_idx_global]) == int(gt_cause[g_idx]):
        #                 attr_hit['cause'] += 1
        #             attr_tot['cause'] += 1
        # 遍历每一帧
        for idx, out in enumerate(results):
            if 'pts_bbox' not in out:
                raise KeyError(f"Missing key 'pts_bbox' in results[{idx}]. "
                            f"simple_test 应返回形如 {{'pts_bbox': {{boxes_3d, scores_3d, labels_3d, attrs?}}}} 的结构。")
            pred = out['pts_bbox']
            boxes_pred = pred['boxes_3d']     # LiDARInstance3DBoxes
            scores_pred = pred['scores_3d']   # (N,)
            labels_pred = pred['labels_3d']   # (N,)
            attrs_pred = pred.get('attrs', None)  # 可选: {'vis': LongTensor[N], 'cause': LongTensor[N]}

            ann = self.get_ann_info(idx)
            boxes_gt = ann['gt_bboxes_3d']    # LiDARInstance3DBoxes
            labels_gt = ann['gt_labels_3d']   # (M,)
            # 可选 GT 属性（若你在 ann_info 里已塞了）
            gt_vis = ann.get('gt_vis_labels', None)          # LongTensor[M]
            gt_cause = ann.get('gt_cause_labels', None)      # LongTensor[M]

            # 逐类评测
            for cid, cname in enumerate(self.CLASSES):
                # 取该类 GT
                gt_mask = (labels_gt == cid)
                num_gt = int(gt_mask.sum().item()) if hasattr(gt_mask, 'sum') else int(gt_mask.sum())
                cls2gt[cname] += num_gt

                # 取该类预测
                pred_mask = (labels_pred == cid)
                if hasattr(pred_mask, 'sum') and int(pred_mask.sum()) == 0:
                    continue

                # 筛选并按分数降序
                if hasattr(scores_pred, 'device'):
                    device = scores_pred.device
                else:
                    device = None
                sel_scores = scores_pred[pred_mask]
                sel_inds = torch.argsort(sel_scores, descending=True)
                sel_scores = sel_scores[sel_inds]

                sel_boxes = boxes_pred.tensor[pred_mask][sel_inds]  # (P, 7)
                # 注意：boxes_gt.tensor 是 (M, 7)
                gt_boxes_t = boxes_gt.tensor[gt_mask] if num_gt > 0 else boxes_gt.tensor.new_zeros((0, 7))

                # 匹配
                if match_mode == 'center':
                    # 2D 中心距离（XY）——稳妥且不依赖 CUDA 自定义算子
                    if sel_boxes.shape[0] == 0:
                        tp = torch.zeros(0, dtype=torch.float32, device=device)
                        fp = torch.zeros(0, dtype=torch.float32, device=device)
                        matches = []
                    else:
                        tp, fp, matches = _match_by_center(sel_boxes[:, :2], gt_boxes_t[:, :2], dist_thr)
                else:
                    # 你后面想切换 BEV IoU / 3D IoU 时，我们再接入相应的 overlaps 函数
                    # 这里先退回 center 匹配，避免环境依赖问题
                    tp, fp, matches = _match_by_center(sel_boxes[:, :2], gt_boxes_t[:, :2], dist_thr)
                    
                        # === 额外统计：真正的 3D IoU（轴对齐） ===
                # matches 里保存的是 (pred_idx_local, gt_idx_local)，只包含 TP
                if len(matches) > 0:
                    # sel_boxes / gt_boxes_t 可能是 Tensor 也可能是 Boxes，统一转成 numpy
                    if hasattr(sel_boxes, 'tensor'):
                        sel_np = sel_boxes.tensor.detach().cpu().numpy()
                    else:
                        sel_np = sel_boxes.detach().cpu().numpy()

                    if hasattr(gt_boxes_t, 'tensor'):
                        gt_np = gt_boxes_t.tensor.detach().cpu().numpy()
                    else:
                        gt_np = gt_boxes_t.detach().cpu().numpy()

                    iou_list = []
                    for p_local, g_local in matches:
                        iou = _axis_aligned_iou_3d_np(sel_np[p_local], gt_np[g_local])
                        iou_list.append(iou)

                    if len(iou_list) > 0:
                        all_matchious.append(np.array(iou_list, dtype=np.float32))


                # 累加 PR 明细（注意每帧要衔接 cumulate，所以只存 per-detection 的 tp/fp 与分数）
                cls2scores[cname].append(sel_scores.detach().cpu())
                cls2tp[cname].append(tp.detach().cpu())
                cls2fp[cname].append(fp.detach().cpu())

                # 属性统计（只在有匹配对、且 pred/gt 都带该属性时计数）
                if do_attr and len(matches) > 0 and attrs_pred is not None:
                    m_pred_mask = pred_mask.nonzero().squeeze(1)[sel_inds]  # 映射回原 preds 索引
                    # gt_mask 是 numpy.bool_ 数组，用 numpy 的 nonzero
                    m_gt_idx = np.nonzero(gt_mask)[0]                 # 映射回原 gt 索引
                    # vis
                    if ('vis' in attrs_pred) and (gt_vis is not None):
                        for p_idx, g_local in matches:
                            g_idx = m_gt_idx[g_local]
                            p_idx_global = m_pred_mask[p_idx]
                            if int(attrs_pred['vis'][p_idx_global]) == int(gt_vis[g_idx]):
                                attr_hit['vis'] += 1
                            attr_tot['vis'] += 1
                    # cause
                    if ('cause' in attrs_pred) and (gt_cause is not None):
                        for p_idx, g_local in matches:
                            g_idx = m_gt_idx[g_local]
                            p_idx_global = m_pred_mask[p_idx]
                            if int(attrs_pred['cause'][p_idx_global]) == int(gt_cause[g_idx]):
                                attr_hit['cause'] += 1
                            attr_tot['cause'] += 1
        
        # 汇总：拼接各帧的 tp/fp/scores，按分数降序累加，算 AP
        results_dict = {}
        aps = []
        for cname in self.CLASSES:
            print(f"{cname}: gt = {cls2gt[cname]}, num_det_frames = {len(cls2scores[cname])}")
            if len(cls2scores[cname]) == 0:
                results_dict[f'AP_{cname}'] = 0.0
                continue
            scores = torch.cat(cls2scores[cname], dim=0)
            tp = torch.cat(cls2tp[cname], dim=0)
            fp = torch.cat(cls2fp[cname], dim=0)

            order = torch.argsort(scores, descending=True)
            tp = tp[order]
            fp = fp[order]

            cum_tp = torch.cumsum(tp, dim=0)
            cum_fp = torch.cumsum(fp, dim=0)
            denom = torch.clamp(cum_tp + cum_fp, min=1)
            prec = (cum_tp / denom).numpy()
            rec = (cum_tp / max(cls2gt[cname], 1)).numpy()
            ap = _voc_ap(rec, prec)
            results_dict[f'AP_{cname}'] = float(ap)
            aps.append(ap)

        results_dict['mAP'] = float(sum(aps) / max(len(aps), 1))

        # 属性指标（若两边都有就给，否则跳过）
        if attr_tot['vis'] > 0:
            results_dict['acc_vis'] = attr_hit['vis'] / max(attr_tot['vis'], 1)
        if attr_tot['cause'] > 0:
            results_dict['acc_cause'] = attr_hit['cause'] / max(attr_tot['cause'], 1)

        if show:
            self.show(results, out_dir)
            # === 汇总全局 matchious mean / max ===
        if len(all_matchious) > 0:
            concat_vals = np.concatenate(all_matchious, axis=0)  # 所有 TP 匹配对
            results_dict['matchious_mean'] = float(concat_vals.mean())
            results_dict['matchious_max']  = float(concat_vals.max())
        else:
            results_dict['matchious_mean'] = 0.0
            results_dict['matchious_max']  = 0.0
        return results_dict

    
    def show(self, results, out_dir):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
        """
        for i, result in enumerate(results):
            example = self.prepare_test_data(i)
            points = example['points'][0]._data.numpy()
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['pts_bbox']['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor
            gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                          Box3DMode.DEPTH)
            pred_bboxes = result['pts_bbox']['boxes_3d'][inds].tensor.numpy()
            pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                            Box3DMode.DEPTH)
            show_result(points, gt_bboxes, pred_bboxes, out_dir, file_name)


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        
        # cls_range_map = eval_configs.class_range
        # radius = np.linalg.norm(box.center[:2], 2)
        # det_range = cls_range_map[classes[box.label]]
        cls_range_map = getattr(eval_configs, 'class_range', {})
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map.get(classes[box.label], float('inf'))
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
