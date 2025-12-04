# 文件：mmdet3d/datasets/pipelines/mine_extra.py
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class RecordNumGT(object):
    def __call__(self, results):
        results['num_gt_before_sample'] = len(results['gt_labels_3d'])
        return results
# 文件：mmdet3d/datasets/pipelines/mine_extra.py（同文件）
@PIPELINES.register_module()
class PadAttrAfterSample(object):
    def __init__(self, ignore_index=-1):
        self.ignore_index = ignore_index

    def __call__(self, results):
        n0 = results.pop('num_gt_before_sample', None)
        n1 = len(results['gt_labels_3d'])
        if n0 is None or n1 <= n0:
            return results

        add = n1 - n0
        def to_np(x, n):
            if x is None:
                return np.full((n,), self.ignore_index, dtype=np.int64)
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            return x

        for key in ['gt_vis_labels', 'gt_cause_labels']:
            arr = results.get(key, None)
            arr = to_np(arr, n0)
            pad = np.full((add,), self.ignore_index, dtype=arr.dtype)
            results[key] = np.concatenate([arr, pad], axis=0)
        return results
@PIPELINES.register_module()
class FetchAttrFromAnnInfo(object):
    """Copy gt_vis_labels / gt_cause_labels from results['ann_info'] to results.
       Always produce arrays of length == len(gt_labels_3d), fill -1 if missing.
    """
    def __init__(self, keys=('gt_vis_labels','gt_cause_labels'), ignore_index=-1):
        self.keys = keys
        self.ignore_index = ignore_index

    def __call__(self, results):
        ann = results.get('ann_info', {})
        n = len(results.get('gt_labels_3d', []))
        for k in self.keys:
            v = ann.get(k, None)
            if v is None:
                arr = np.full((n,), self.ignore_index, dtype=np.int64)
            else:
                if isinstance(v, torch.Tensor):
                    v = v.cpu().numpy()
                v = np.asarray(v, dtype=np.int64)
                # 某些实现里 ann_info 是 mask 前的数组，这里以当前 gt 数量为准裁/补
                if v.shape[0] != n:
                    if v.shape[0] > n:
                        arr = v[:n]
                    else:
                        pad = np.full((n - v.shape[0],), self.ignore_index, dtype=np.int64)
                        arr = np.concatenate([v, pad], axis=0)
                else:
                    arr = v
            results[k] = arr
        return results