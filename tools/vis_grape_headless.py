import os
import os.path as osp
from mmcv.parallel import MMDataParallel, DataContainer
import sys
import numpy as np
import torch
import open3d as o3d
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_detector


def load_points_from_bin(bin_path, load_dim=4, use_dim=(0, 1, 2)):
    """从 nuScenes/KITTI 风格 .bin 读取点云，并返回 xyz."""
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, load_dim)
    return pts[:, use_dim]


def boxes_to_lineset(corners, color=(1.0, 0.0, 0.0)):
    """将 box 角点 (N, 8, 3) 转成 Open3D LineSet，用于画线框."""
    corners = np.asarray(corners)  # (N, 8, 3)
    all_pts = []
    all_lines = []
    all_colors = []

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
    ]

    offset = 0
    for box_corners in corners:
        all_pts.extend(box_corners.tolist())
        for e in edges:
            all_lines.append([offset + e[0], offset + e[1]])
            all_colors.append(list(color))
        offset += 8

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(all_pts))
    line_set.lines = o3d.utility.Vector2iVector(
        np.asarray(all_lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.asarray(all_colors))
    return line_set

def boxes_to_wireframe_points(corners, color=(1.0, 0.0, 0.0), num_per_edge=60):
    """将 box 角点 (N, 8, 3) 转成密一点的线框点云."""
    corners = np.asarray(corners)  # (N, 8, 3)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
    ]

    pts_list = []
    cols_list = []
    color = np.array(color, dtype=np.float32)

    for box in corners:
        for i, j in edges:
            p0 = box[i]
            p1 = box[j]
            # 在每条边上均匀采样 num_per_edge 个点
            alphas = np.linspace(0.0, 1.0, num_per_edge, dtype=np.float32)[:, None]  # (K,1)
            seg = (1 - alphas) * p0[None, :] + alphas * p1[None, :]  # (K,3)
            pts_list.append(seg)
            cols_list.append(np.tile(color[None, :], (num_per_edge, 1)))

    if len(pts_list) == 0:
        return None

    pts = np.concatenate(pts_list, axis=0)
    cols = np.concatenate(cols_list, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def export_one_sample(dataset, idx, result, out_dir, score_thr=0.2):
    """导出一帧的点云 + GT box + 预测 box（全部写 PLY，不创建窗口）."""
    # 一定要走 get_data_info，里面才会构造 pts_filename
    info = dataset.get_data_info(idx)
    pts_file = info['pts_filename']

    # 1) 读取原始点云（只要 xyz 就够可视化了）
    points_xyz = load_points_from_bin(
        pts_file, load_dim=5, use_dim=(0, 1, 2))

    # 2) 解析预测结果
    #    result: list，batch size = 1，所以取 result[0]
    pred = result[0]

    boxes_3d = None
    scores_3d = None

    if isinstance(pred, dict):
        if 'boxes_3d' in pred:
            boxes_3d = pred['boxes_3d']
            scores_3d = pred.get('scores_3d', None)
        elif 'pts_bbox' in pred:
            boxes_3d = pred['pts_bbox']['boxes_3d']
            scores_3d = pred['pts_bbox'].get('scores_3d', None)
    else:
        # 某些版本可能直接返回 boxes_3d 对象
        boxes_3d = getattr(pred, 'boxes_3d', None)
        scores_3d = getattr(pred, 'scores_3d', None)

    if boxes_3d is not None:
        if scores_3d is not None:
            scores = scores_3d.detach().cpu().numpy()
            keep = scores >= score_thr
            if keep.sum() > 0:
                boxes_3d = boxes_3d[keep]
            else:
                boxes_3d = boxes_3d[:0]  # 空
        pred_corners = boxes_3d.corners.detach().cpu().numpy()
    else:
        pred_corners = np.zeros((0, 8, 3), dtype=np.float32)

    if boxes_3d is not None and scores_3d is not None:
        scores = scores_3d.detach().cpu().numpy()
        order = scores.argsort()[::-1]
        print(f"sample {idx}: top scores:", scores[order[:5]])
    # 3) GT boxes
    ann = dataset.get_ann_info(idx)
    gt_boxes = ann.get('gt_bboxes_3d', None)
    gt_labels = ann.get('gt_labels_3d', None)

    if gt_boxes is not None and len(gt_boxes) > 0:
        gt_corners = gt_boxes.corners.detach().cpu().numpy()
    else:
        gt_corners = np.zeros((0, 8, 3), dtype=np.float32)

    # 3.1) 只取 grape 的 GT boxes（如果数据集里有 'grape' 这个类）
    grape_only_corners = np.zeros((0, 8, 3), dtype=np.float32)
    if gt_corners.shape[0] > 0 and gt_labels is not None:
        # gt_labels_3d 可能是 tensor 也可能是 numpy
        if isinstance(gt_labels, torch.Tensor):
            gt_labels_np = gt_labels.cpu().numpy()
        else:
            gt_labels_np = np.asarray(gt_labels, dtype=np.int64)

        class_names = getattr(dataset, 'CLASSES', None)
        if class_names is not None and 'grape' in class_names:
            grape_idx = class_names.index('grape')
            mask = (gt_labels_np == grape_idx)
            grape_only_corners = gt_corners[mask]
        else:
            # 如果只有单类，或者没找到 'grape'，那就全当作 grape
            grape_only_corners = gt_corners

    # 4) 用 Open3D 写文件（只用 I/O，不创建可视化窗口）
    os.makedirs(out_dir, exist_ok=True)

    # 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_path = osp.join(out_dir, f"sample_{idx:04d}_points.ply")
    o3d.io.write_point_cloud(pcd_path, pcd)

    # 预测 box：红色 —— 用密集点线框
    if pred_corners.shape[0] > 0:
        pred_wire = boxes_to_wireframe_points(
            pred_corners, color=(1.0, 0.0, 0.0), num_per_edge=60
        )
        if pred_wire is not None:
            pred_path = osp.join(out_dir, f"sample_{idx:04d}_pred_boxes.ply")
            o3d.io.write_point_cloud(pred_path, pred_wire)

    # GT box：绿色（所有类别）—— 用密集点线框
    if gt_corners.shape[0] > 0:
        gt_wire = boxes_to_wireframe_points(
            gt_corners, color=(0.0, 1.0, 0.0), num_per_edge=60
        )
        if gt_wire is not None:
            gt_path = osp.join(out_dir, f"sample_{idx:04d}_gt_boxes.ply")
            o3d.io.write_point_cloud(gt_path, gt_wire)

    # 只画 grape 的 GT box：蓝色 —— 用密集点线框
    if grape_only_corners.shape[0] > 0:
        grape_wire = boxes_to_wireframe_points(
            grape_only_corners, color=(0.0, 0.0, 1.0), num_per_edge=60
        )
        if grape_wire is not None:
            grape_path = osp.join(
                out_dir, f"sample_{idx:04d}_gt_grape_boxes.ply")
            o3d.io.write_point_cloud(grape_path, grape_wire)

    print(f"[OK] exported sample {idx} -> {out_dir}")


def main():
    # === 根据你当前的 config / ckpt 路径修改这两行 ===
    config_file = "configs/transfusion_minegrape_L_grapeonly.py"
    checkpoint_file = "work_dirs/minegrape_go_L/epoch_100.pth"
    out_dir = "vis_grape_headless"
    max_samples = 5   # 想导出几帧自己改
    os.makedirs(out_dir, exist_ok=True)
    cfg = Config.fromfile(config_file)
    cfg.data.test.test_mode = True

    # 构建 test 数据集
    dataset = build_dataset(cfg.data.test)

    # 构建模型（模仿 tools/test.py）
    cfg.model.pretrained = None
    if cfg.get('plugin', False):
        # 兼容某些自带 plugin 的分支（可选）
        import importlib
        plugin_dir = cfg.get('plugin_dir', None)
        if plugin_dir is not None:
            _module_dir = osp.dirname(plugin_dir)
            _module_name = osp.basename(plugin_dir)
            sys.path.insert(0, _module_dir)
            importlib.import_module(_module_name)

    model = build_detector(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    # 加载权重
    checkpoint = load_checkpoint(
        model, checkpoint_file, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if torch.cuda.is_available():
        model = MMDataParallel(model.cuda(), device_ids=[0])
    else:
        # 没 GPU 的话就放 CPU，上 MMDataParallel 主要是为了处理 DataContainer
        model = MMDataParallel(model, device_ids=[-1])
    model.eval()

    # 构建 dataloader（直接复用官方单卡测试的方式）
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)

    indices_to_save = set(range(min(max_samples, len(dataset))))
    print(f"Will export indices: {sorted(list(indices_to_save))}")

    for i, data in enumerate(data_loader):
        if i not in indices_to_save:
            continue

        # 把 data 移到 device（build_dataloader 已经处理好了 DataContainer）
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        export_one_sample(dataset, i, result, out_dir, score_thr=0.35)

        indices_to_save.remove(i)
        if not indices_to_save:
            break


if __name__ == "__main__":
    main()
