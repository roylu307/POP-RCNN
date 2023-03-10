from .pvrcnn_head import PVRCNNHead
from .voxelrcnn_head import VoxelRCNNHead
from .roi_head_template import RoIHeadTemplate

from .pophead_V import PGHead_V


__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PVRCNNHead': PVRCNNHead,
    'VoxelRCNNHead': VoxelRCNNHead,
    'PGHead_V': PGHead_V
}
