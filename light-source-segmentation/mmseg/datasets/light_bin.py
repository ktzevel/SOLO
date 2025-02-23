# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class LightDatasetBin(BaseSegDataset):
    """Light dataset for binary classification.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('unlabeled', 'front_light'),
        palette=[[128, 64, 128], [107, 142, 35]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_label.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

