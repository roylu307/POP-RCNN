from .detector3d_template import Detector3DTemplate

from .pv_rcnn import PVRCNN

from .voxel_rcnn import VoxelRCNN
from .pv_rcnn_plusplus import PVRCNNPlusPlus


__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'PVRCNN': PVRCNN,
    'VoxelRCNN': VoxelRCNN,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
