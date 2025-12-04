import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class TransFusionDetector(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self, **kwargs):
        super(TransFusionDetector, self).__init__(**kwargs)

        self.freeze_img = kwargs.get('freeze_img', True)
        self.init_weights(pretrained=kwargs.get('pretrained', None))

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(TransFusionDetector, self).init_weights(pretrained)

        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
                # ---- 新增：统一 img_metas 的格式 ----
            # 训练时是 List[Dict]，但是在 forward_test / simple_test 里
            # 会传进来一个 Dict（来自 img_metas[0]），这里统一成 List[Dict]，
            # 防止后面 for 循环迭代到的是字符串 key。
            if isinstance(img_metas, dict):
                img_metas = [img_metas]
            elif isinstance(img_metas, (list, tuple)) and len(img_metas) > 0 \
                    and isinstance(img_metas[0], (list, tuple)):
                # 兼容偶尔会出现的 [[meta]] 这种嵌套一层的情况
                img_metas = img_metas[0]
                # ---- 新增结束 ----
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img.float())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
            """Accept Tensor (N,C) or List[Tensor]; return concatenated tensors.
            Returns:
                voxels:      (M, max_points, C)  float
                num_points:  (M,)               int
                coors:       (M, 4)             long  [batch_idx, z, y, x]
            """
            if isinstance(points, torch.Tensor):
                points = [points]
            elif not isinstance(points, (list, tuple)):
                if hasattr(points, 'tensor'):
                    points = [points.tensor]
                else:
                    points = [points]

            voxels_list, num_points_list, coors_list = [], [], []

            # -------- 新增：对每个样本做最多 N 点的随机下采样（例如 300k）--------
            POINT_CAP = 200000
            for i, res in enumerate(points):
                if res.shape[0] > POINT_CAP:
                    idx = torch.randint(0, res.shape[0], (POINT_CAP,), device=res.device)
                    res = res.index_select(0, idx)
                # --------------------------------------------------------------

                v, c, n = self.pts_voxel_layer(res)  # v:(m,T,C) c:(m,3) n:(m,)
                voxels_list.append(v); num_points_list.append(n); coors_list.append(c)

            voxels = torch.cat(voxels_list, dim=0)
            num_points = torch.cat(num_points_list, dim=0)
            coors_w_batch = []
            for i, coor in enumerate(coors_list):
                b = torch.full((coor.shape[0], 1), i, dtype=coor.dtype, device=coor.device)
                coors_w_batch.append(torch.cat([b, coor], dim=1))
            coors = torch.cat(coors_w_batch, dim=0)
            return voxels, num_points, coors

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_vis_labels=None,           # ← 新增
                      gt_cause_labels=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore,
                                                gt_vis_labels=gt_vis_labels,gt_cause_labels=gt_cause_labels)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        #--------------------------------------------------------------------------------------        
        
        #--------------------------------------------------------------------------------------  
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          gt_vis_labels=None, gt_cause_labels=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        # loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        # losses = self.pts_bbox_head.loss(*loss_inputs)  
        losses = self.pts_bbox_head.loss(
            gt_bboxes_3d, gt_labels_3d, outs,
            gt_vis_labels_list=gt_vis_labels,             # ← 新增
            gt_cause_labels_list=gt_cause_labels,         # ← 新增
            img_metas=img_metas)
        return losses

    # transfusion.py
    def simple_test_pts(self, x, x_img, img_metas, rescale=False, datrain=0):
        """Return list of length B; each item is a dict:
           {
             'boxes_3d': ...,
             'scores_3d': ...,
             'labels_3d': ...,
             'attrs': {'vis': ..., 'cause': ...}  # 可选
           }
        """

        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale
        )

        bbox_results = []
        for out in bbox_list:
            boxes_3d  = out['boxes_3d']
            scores_3d = out['scores_3d']
            labels_3d = out['labels_3d']

            res = dict(
                boxes_3d=boxes_3d,
                scores_3d=scores_3d,
                labels_3d=labels_3d,
            )

            # ★ 把辅助属性打包成 attrs，后面 evaluate 会按这个约定读取
            attrs = {}
            if out.get('attr_vis', None) is not None:
                attrs['vis'] = out['attr_vis']
            if out.get('attr_cause', None) is not None:
                attrs['cause'] = out['attr_cause']
            if len(attrs) > 0:
                res['attrs'] = attrs

            bbox_results.append(res)

        return bbox_results


    def simple_test(self, points, img_metas, img=None, rescale=False, datrain=0):
        """Return list[ dict(pts_bbox=...) ] of length B."""
        # ---- 新增：统一 img_metas 的格式 ----
        # 训练时是 List[Dict]，forward_test 里传进来的则是单个 Dict（img_metas[0]）。
        # 这里统一成 List[Dict]，以便后续 pts_bbox_head 和 head.forward_single
        # 一律按 List[Dict] 使用。
        if isinstance(img_metas, dict):
            # 来自 Base3DDetector.forward_test: img_metas[0]
            img_metas = [img_metas]
        elif isinstance(img_metas, (list, tuple)) and len(img_metas) > 0 \
                and isinstance(img_metas[0], (list, tuple)):
            # 兼容 [[meta]] 这种嵌套一层的情况
            img_metas = img_metas[0]
        # ---- 新增结束 ----
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        has_pts = pts_feats is not None and len(pts_feats) > 0

        results = []
        if has_pts:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas,
                rescale=rescale, datrain=datrain)

            # 现在 bbox_pts 是 list[dict]，每个 dict 已经包含 boxes_3d/scores_3d/labels_3d/attrs
            for res in bbox_pts:
                results.append({'pts_bbox': res})

        else:
            # 理论上不会走到；兜底返回空预测，但保持 batch 对齐
            B = len(points) if isinstance(points, (list, tuple)) else 1
            box_type_3d = img_metas[0]['box_type_3d']
            device = points[0].device if isinstance(points, (list, tuple)) else points.device
            for _ in range(B):
                empty_boxes = box_type_3d(
                    torch.zeros((0, 7), device=device),
                    box_dim=7
                )
                results.append({
                    'pts_bbox': dict(
                        boxes_3d=empty_boxes,
                        scores_3d=torch.empty(0, device=device),
                        labels_3d=torch.empty(0, dtype=torch.long, device=device),
                    )
                })

        return results
