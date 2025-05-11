import warnings

from mmdet.models.builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                                  ROI_EXTRACTORS, SHARED_HEADS)
from .registry import FUSION_LAYERS, MIDDLE_ENCODERS, VOXEL_ENCODERS

from mmcv.utils import Registry, build_from_cfg

def build(cfg, registry, default_args=None):
    """Build a module.
    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list): # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg # build_from_cfg()在后面进行讲解。
        ]
        return nn.Sequential(*modules) # nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)


def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    """Build RoI feature extractor."""
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    """Build shared head of detector."""
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)


def build_loss(cfg):
    """Build loss function."""
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_voxel_encoder(cfg):
    """Build voxel encoder."""
    return build(cfg, VOXEL_ENCODERS)


def build_middle_encoder(cfg):
    """Build middle level encoder."""
    return build(cfg, MIDDLE_ENCODERS)


def build_fusion_layer(cfg):
    """Build fusion layer."""
    return build(cfg, FUSION_LAYERS)
