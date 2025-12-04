
from mmdet.datasets import DATASETS

from .nuscenes_dataset import NuScenesDataset


@DATASETS.register_module()
class NuScenesGrapeDataset(NuScenesDataset):
    """只用于单类 grape 检测的版本。
    其它逻辑（evaluate、属性统计等）全部沿用 NuScenesDataset。
    """
    # 单类
    CLASSES = ('grape',)

    def __init__(self, *args, **kwargs):
        # 如果 config 里没有显式传 classes，就默认用 ('grape',)
        if 'classes' not in kwargs or kwargs['classes'] is None:
            kwargs['classes'] = self.CLASSES
        super().__init__(*args, **kwargs)
