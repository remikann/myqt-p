# tools/vis_bev_any.py  —— 只看 results.pkl，递归解析，BEV 可视化（Matplotlib/Agg）
import os, os.path as osp
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv

def to_numpy(x):
    if x is None: return None
    if hasattr(x, 'detach'): return x.detach().cpu().numpy()
    return np.asarray(x)

def corners_bev(boxes3d):
    """(N,4,2) XY 底面四角"""
    # 先用对象自带
    if hasattr(boxes3d, 'bottom_corners'):
        bc = boxes3d.bottom_corners  # (N,3,4)
        return bc[:, :2, :].transpose(0, 2, 1)
    if hasattr(boxes3d, 'corners'):
        c = boxes3d.corners  # (N,8,3)
        return c[:, [0,1,2,3], :2]
    # 兜底：按 (x,y,z,w,l,h,yaw)
    t = boxes3d.tensor if hasattr(boxes3d, 'tensor') else boxes3d
    t = to_numpy(t)
    if t is None or t.size == 0: return np.zeros((0,4,2), dtype=np.float32)
    outs = np.zeros((t.shape[0],4,2), dtype=np.float32)
    for i in range(t.shape[0]):
        x,y,z,w,l,h,yaw = t[i,:7]
        dx, dy = l/2, w/2
        cs, sn = np.cos(yaw), np.sin(yaw)
        rect = np.array([[ dx,  dy],[ dx, -dy],[-dx, -dy],[-dx,  dy]], dtype=np.float32)
        R = np.array([[cs,-sn],[sn,cs]], dtype=np.float32)
        outs[i] = rect @ R.T + np.array([x,y])
    return outs

BOX_KEYS   = ('boxes_3d','bboxes_3d','boxes')
SCORE_KEYS = ('scores_3d','scores')
LABEL_KEYS = ('labels_3d','labels')
NESTED_KEYS= ('pts_bbox','pred_instances_3d','results','outputs','data')

def _try_build(m):
    boxes=scores=labels=None
    for k in BOX_KEYS:
        if isinstance(m, dict) and k in m: boxes=m[k]; break
        if hasattr(m,k): boxes=getattr(m,k); break
    for k in SCORE_KEYS:
        if isinstance(m, dict) and k in m: scores=m[k]; break
        if hasattr(m,k): scores=getattr(m,k); break
    for k in LABEL_KEYS:
        if isinstance(m, dict) and k in m: labels=m[k]; break
        if hasattr(m,k): labels=getattr(m,k); break
    if boxes is not None and scores is not None:
        return dict(boxes_3d=boxes, scores_3d=scores, labels_3d=labels)
    return None

def _unwrap_one(x):
    changed=True
    while changed:
        changed=False
        if isinstance(x,(list,tuple)) and len(x)==1:
            x=x[0]; changed=True
        elif isinstance(x,dict):
            for k in ('data','result','results','outputs'):
                if k in x and isinstance(x[k],(list,tuple)) and len(x[k])==1:
                    x=x[k][0]; changed=True; break
    return x

def extract_triplet(obj, depth=0, max_depth=8):
    if obj is None or depth>max_depth: return None
    obj=_unwrap_one(obj)
    got=_try_build(obj)
    if got is not None: return got
    if isinstance(obj,dict):
        for k in NESTED_KEYS:
            if k in obj:
                got=extract_triplet(obj[k], depth+1, max_depth)
                if got is not None: return got
    if isinstance(obj,(list,tuple)):
        if len(obj)>=2:
            b,s=(obj[0], obj[1]); l=(obj[2] if len(obj)>=3 else None)
            # 粗判：像 boxes 就收
            if hasattr(b,'tensor') or (hasattr(b,'shape') and len(b.shape)>=2 and b.shape[-1]>=7):
                return dict(boxes_3d=b, scores_3d=s, labels_3d=l)
        for it in obj:
            got=extract_triplet(it, depth+1, max_depth)
            if got is not None: return got
    if isinstance(obj,dict):
        for v in obj.values():
            got=_try_build(v)
            if got is not None: return got
        for v in obj.values():
            got=extract_triplet(v, depth+1, max_depth)
            if got is not None: return got
    for k in NESTED_KEYS+BOX_KEYS+SCORE_KEYS+LABEL_KEYS:
        if hasattr(obj,k):
            got=extract_triplet(getattr(obj,k), depth+1, max_depth)
            if got is not None: return got
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--score-thr', type=float, default=0.0)
    ap.add_argument('--xlim', type=float, nargs=2, default=[-54,54])
    ap.add_argument('--ylim', type=float, nargs=2, default=[-54,54])
    args=ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results=mmcv.load(args.results)
    # 统一成 list
    if isinstance(results, dict):
        for k in ('results','outputs','data'):
            if k in results and isinstance(results[k], (list,tuple)):
                results = results[k]; break
        else:
            results = [results]

    total=len(results)
    print(f'[INFO] results items: {total}')
    for i,item in enumerate(results):
        det=extract_triplet(item)
        fig=plt.figure(figsize=(6,6), dpi=150); ax=plt.gca()
        detn=0
        if det is not None:
            scores=det['scores_3d']; boxes=det['boxes_3d']
            scores_np=to_numpy(scores)
            mask = (scores_np > args.score_thr) if scores_np is not None else slice(None)
            try:
                boxes_sel = boxes[mask]
                if hasattr(boxes_sel,'tensor'):
                    if boxes_sel.tensor.shape[0]>0:
                        bev=corners_bev(boxes_sel); detn=bev.shape[0]
                        for k in range(detn):
                            poly=np.vstack([bev[k], bev[k,0:1]])
                            ax.plot(poly[:,0], poly[:,1], linewidth=1.0)
                            c=bev[k].mean(axis=0); v=bev[k,0]-bev[k,3]
                            v = v/(np.linalg.norm(v)+1e-6)*1.0
                            ax.arrow(c[0],c[1],v[0],v[1], head_width=0.5, length_includes_head=True)
            except Exception as e:
                print(f'[WARN] draw failed at idx {i}: {e}')
        else:
            if i==0: print('[WARN] cannot parse detection at idx 0; still output empty canvas.')

        ax.set_xlim(args.xlim); ax.set_ylim(args.ylim); ax.set_aspect('equal')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.set_title(f'idx_{i:04d} (thr={args.score_thr}, det={detn})')
        out_png=osp.join(args.out, f'idx_{i:04d}.png')
        fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)
    print('[INFO] done.')

if __name__=='__main__':
    main()
