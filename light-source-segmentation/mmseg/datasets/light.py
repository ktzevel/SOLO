# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class LightDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
                                    # id
        classes=('unlabeled'        # 0
               , 'window_building'  # 1
               , 'window_parked'    # 2
               , 'window_transport' # 3
               , 'traffic_light'    # 4
               , 'street_light'     # 5
               , 'front_light'      # 6
               , 'rear_light'       # 7
               , 'advertisement'    # 8
               , 'clock_inferred'   # 9
               ),

        palette=[[128, 64, 128]     # 0
               , [244, 35, 232]     # 1
               , [70, 70, 70]       # 2
               , [102, 102, 156]    # 3
               , [190, 153, 153]    # 4
               , [153, 153, 153]    # 5
               , [250, 170, 30]     # 6
               , [220, 220, 0]      # 7
               , [107, 142, 35]     # 8
               , [152, 251, 152]    # 9
               ])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_label.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
