# tools/inspect_init_and_freeze_v2.py
import sys
from mmcv import Config
from mmdet3d.models import build_detector
from mmcv.runner import load_checkpoint, build_optimizer

def main(cfg_path):
    cfg = Config.fromfile(cfg_path)
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    if cfg.get('load_from', None):
        load_checkpoint(model, cfg.load_from, map_location='cpu')

    # 统计 requires_grad（和你原来的脚本一致）
    def count_by_prefix(prefix, pred):
        return sum(p.numel() for n,p in model.named_parameters() if pred(n,p) and n.startswith(prefix))

    print('requires_grad=True params:')
    for k in ['img_backbone','img_neck','pts_voxel_encoder','pts_middle_encoder','pts_backbone','pts_neck','pts_bbox_head']:
        v = count_by_prefix(k, lambda n,p: p.requires_grad)
        print(f'  {k:<20} -> {v}')

    # 关键：构建 optimizer，看 lr>0 的参数
    optimizer = build_optimizer(model, cfg.optimizer)
    # 把所有 lr>0 的参数名收集起来（只看第一步的初始 lr）
    lr_pos = set()
    for g in optimizer.param_groups:
        if g['lr'] > 0:
            for p in g['params']:
                lr_pos.add(id(p))

    print('\nlr>0 params (what will actually update):')
    for k in ['img_backbone','img_neck','pts_voxel_encoder','pts_middle_encoder','pts_backbone','pts_neck','pts_bbox_head']:
        v = count_by_prefix(k, lambda n,p: id(p) in lr_pos)
        print(f'  {k:<20} -> {v}')

if __name__ == '__main__':
    main(sys.argv[1])
