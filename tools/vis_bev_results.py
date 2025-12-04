# tools/vis_bev_results.py  —— 极简稳定版（只支持 {'pts_bbox': {boxes_3d,scores_3d,labels_3d}}）
import os, os.path as osp
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset

def corners_bev(boxes3d):
    """Return (N,4,2) bottom corners (XY) for LiDAR boxes."""
    # 优先用对象方法
    if hasattr(boxes3d, 'bottom_corners'):
        bc = boxes3d.bottom_corners  # (N,3,4)
        return bc[:, :2, :].transpose(0, 2, 1)
    if hasattr(boxes3d, 'corners'):
        c = boxes3d.corners  # (N,8,3)
        return c[:, [0,1,2,3], :2]
    # 兜底：按 (x,y,z,w,l,h,yaw) 近似生成
    t = boxes3d.tensor if hasattr(boxes3d, 'tensor') else boxes3d
    t = t.detach().cpu().numpy()
    N = t.shape[0]
    outs = np.zeros((N,4,2), dtype=np.float32)
    for i in range(N):
        x,y,z,w,l,h,yaw = t[i,:7]
        dx, dy = l/2, w/2
        cs, sn = np.cos(yaw), np.sin(yaw)
        rect = np.array([[ dx,  dy],[ dx, -dy],[-dx, -dy],[-dx,  dy]], dtype=np.float32)
        R = np.array([[cs,-sn],[sn,cs]], dtype=np.float32)
        outs[i] = rect @ R.T + np.array([x,y])
    return outs

def load_points_bin(bin_path, use_dim=3, pc_range=None, max_pts=None):
    arr = np.fromfile(bin_path, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0,3), dtype=np.float32)
    if arr.size % 5 == 0:
        pts = arr.reshape(-1,5)[:, :use_dim]
    elif arr.size % 4 == 0:
        pts = arr.reshape(-1,4)[:, :use_dim]
    else:
        pts = arr.reshape(-1, use_dim)[:, :use_dim]
    if pc_range is not None and pts.shape[0] > 0:
        x1,y1,z1,x2,y2,z2 = pc_range
        m = (pts[:,0]>=x1)&(pts[:,0]<=x2)&(pts[:,1]>=y1)&(pts[:,1]<=y2)
        pts = pts[m]
    if max_pts is not None and pts.shape[0] > max_pts:
        idx = np.random.choice(pts.shape[0], max_pts, replace=False)
        pts = pts[idx]
    return pts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--score-thr', type=float, default=0.1)
    ap.add_argument('--max-pts', type=int, default=200000)
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    if 'test' in cfg.data:
        cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    results = mmcv.load(args.results)
    # results 可能是 list 或 dict 包了一层
    if isinstance(results, dict):
        for k in ('results','outputs','data'):
            if k in results:
                results = results[k]
                break

    os.makedirs(args.out, exist_ok=True)
    pc_range = cfg.get('point_cloud_range', [-54,-54,-5,54,54,3])
    x1,y1,_,x2,y2,_ = pc_range

    N = min(len(results), len(getattr(dataset, 'data_infos', [])) or len(results))
    for i in range(N):
        info = dataset.data_infos[i] if hasattr(dataset, 'data_infos') else {}
        lidar_path = info.get('lidar_path', info.get('data_path', None))
        token = info.get('token', osp.splitext(osp.basename(lidar_path or f'{i:06d}'))[0])

        # —— 关键：按你输出的固定结构读取 —— #
        item = results[i]
        assert isinstance(item, dict) and 'pts_bbox' in item, 'unexpected result format'
        d = item['pts_bbox']
        boxes = d['boxes_3d']
        scores = d['scores_3d']
        labels = d.get('labels_3d', None)

        # 统一成 numpy 做阈值
        if hasattr(scores, 'detach'):
            scores_np = scores.detach().cpu().numpy()
        else:
            scores_np = np.asarray(scores)
        mask = scores_np > args.score_thr
        boxes_sel = boxes[mask]

        # 读取点云（可选）
        if lidar_path and osp.isfile(lidar_path):
            pts = load_points_bin(lidar_path, use_dim=3, pc_range=pc_range, max_pts=args.max_pts)
        else:
            pts = np.zeros((0,3), dtype=np.float32)

        # 画图
        fig = plt.figure(figsize=(6,6), dpi=150)
        ax = plt.gca()
        if pts.shape[0] > 0:
            ax.scatter(pts[:,0], pts[:,1], s=0.1, alpha=0.5)

        detn = 0
        if hasattr(boxes_sel, 'tensor') and boxes_sel.tensor.shape[0] > 0:
            bev = corners_bev(boxes_sel)
            detn = bev.shape[0]
            for k in range(detn):
                poly = np.vstack([bev[k], bev[k,0:1]])
                ax.plot(poly[:,0], poly[:,1], linewidth=1.0)
                c = bev[k].mean(axis=0)
                v = bev[k,0] - bev[k,3]
                v = v / (np.linalg.norm(v) + 1e-6) * 1.0
                ax.arrow(c[0], c[1], v[0], v[1], head_width=0.5, length_includes_head=True)

        ax.set_xlim([x1,x2]); ax.set_ylim([y1,y2]); ax.set_aspect('equal')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'{token} (thr={args.score_thr}, det={detn})')
        fig.savefig(osp.join(args.out, f'{token}.png'), bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    main()
