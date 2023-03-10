import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from collections import OrderedDict
from typing import List, Callable, Optional, Union, Tuple
import numpy as np

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils
# from mmcv.cnn import build_norm_layer, build_conv_layer

# from timm import create_model
# from timm.models.layers import create_conv2d, create_pool2d, Swish, get_act_layer

# from ..utils import get_fpn_config, _init_weight_alt, _init_weight

_DEBUG = False


def make_fc_layer(input_channels, output_channels, dropout=0):
    fc_layer = []

    fc_layer.extend([
        nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False),
        nn.BatchNorm1d(output_channels),
        nn.ReLU()
    ])

    if dropout >= 0 and k == 0:
        fc_layer.append(nn.Dropout(dropout))
    fc_layer = nn.Sequential(*fc_layers)
    return fc_layer

def make_fc_layers(input_channels, output_channels, fc_list, dropout=0):
    fc_layers = []
    pre_channel = input_channels
    for k in range(0, fc_list.__len__()):
        fc_layers.extend([
            nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
            nn.BatchNorm1d(fc_list[k]),
            nn.ReLU()
        ])
        pre_channel = fc_list[k]
        if dropout >= 0 and k == 0:
            fc_layers.append(nn.Dropout(dropout))
    fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
    fc_layers = nn.Sequential(*fc_layers)
    return fc_layers


class Interpolate3NN(nn.Module):
    def __init__(self):
        """
        Args:
            mlp: list of int
        """
        super().__init__()
        # shared_mlps = []
        # for k in range(len(mlp) - 1):
        #     shared_mlps.extend([
        #         nn.Conv2d(mlp[k], mlp[k + 1], kernel_size=1, bias=False),
        #         nn.BatchNorm2d(mlp[k + 1]),
        #         nn.ReLU()
        #     ])
        # self.mlp = nn.Sequential(*shared_mlps)

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features):
        """
        Args:
            new_xyz: (N1 + N2 ..., 3)
            known: (M1 + M2 ..., 3)
            unknow_feats: (N1 + N2 ..., C1)
            known_feats: (M1 + M2 ..., C2)

        Returns:
            new_features: (N1 + N2 ..., C_out)
        """
        dist, idx = pointnet2_utils.three_nn(new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=-1, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(features, idx, weight)

        # if unknown_feats is not None:
        #     new_features = torch.cat([interpolated_feats, unknown_feats], dim=1)  # (N1 + N2 ..., C2 + C1)
        # else:
        #     new_features = interpolated_feats
        # new_features = new_features.permute(1, 0)[None, :, :, None]  # (1, C, N1 + N2 ..., 1)
        # new_features = self.mlp(new_features)

        # new_features = new_features.squeeze(dim=0).squeeze(dim=-1).permute(1, 0)  # (N1 + N2 ..., C)
        return interpolated_feats


class ResampleFeatureMap(nn.Module):

    def __init__(
            self, SA_cfg, in_channels, out_channels, reduction_ratio=1.):
        super(ResampleFeatureMap, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        # self.conv_after_downsample = conv_after_downsample
        self.SA_cfg = SA_cfg
        conv = None
        if in_channels != out_channels:
            conv = make_fc_layer(in_channels, out_channels)

        self.num_groups = self.SA_cfg.NUM_GROUPS
        self.layers = []
        c_in = 0

        if reduction_ratio != 1:
            cur_config = self.SA_cfg.GROUP_CFG_0
            # self.vector_pool_module = pointnet2_stack_modules.VectorPoolAggregationModule_NoConv(
            #     input_channels=in_channels, num_local_voxel=cur_config.NUM_LOCAL_VOXEL,
            #     post_mlps=[out_channels],
            #     max_neighbor_distance=cur_config.MAX_NEIGHBOR_DISTANCE,
            #     neighbor_nsample=cur_config.NEIGHBOR_NSAMPLE,
            #     local_aggregation_type=self.SA_cfg.LOCAL_AGGREGATION_TYPE,
            #     num_reduced_channels=self.SA_cfg.get('NUM_REDUCED_CHANNELS', None),
            #     num_channels_of_local_aggregation=self.SA_cfg.NUM_CHANNELS_OF_LOCAL_AGGREGATION,
            #     neighbor_distance_multiplier=2.0, from_in
            # )

            self.vector_pool_module = Interpolate3NN()

        # if self.local_aggregation_type == 'local_interpolation':
        #     self.local_interpolate_module = pointnet2_stack_modules.VectorPoolLocalInterpolateModule(
        #         mlp=None, num_voxels=self.num_local_voxel,
        #         max_neighbour_distance=self.max_neighbour_distance,
        #         nsample=self.neighbor_nsample,
        #         neighbor_type=self.neighbor_type,
        #         neighbour_distance_multiplier=neighbor_distance_multiplier,
        #     )

        else:
            if conv is not None:
                self.conv = conv

    def forward(self, xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features):

        # voxel_centers = self.get_dense_voxels_by_center(
        #     point_centers=new_xyz, max_neighbour_distance=self.max_neighbour_distance, num_voxels=self.num_local_voxel
        # )  # (M1 + M2 + ..., total_voxels, 3)
        # voxel_features = self.local_interpolate_module.forward(
        #     support_xyz=xyz, support_features=features, xyz_batch_cnt=xyz_batch_cnt,
        #     new_xyz=new_xyz, new_xyz_grid_centers=voxel_centers, new_xyz_batch_cnt=new_xyz_batch_cnt
        # )  # ((M1 + M2 ...) * total_voxels, C)

        # voxel_features = voxel_features.contiguous().view(-1, self.total_voxels * voxel_features.shape[-1])

        if self.reduction_ratio != 1:
            new_features = self.vector_pool_module(xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features)
            x = new_features
        else:
            x = features
        return x


class FpnCombine_same_grid(nn.Module):
    def __init__(self, feature_info, fpn_config, neck_cfg, fpn_channels, inputs_offsets,
                target_reduction, weight_method='attn', ):
        super(FpnCombine_same_grid, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method
        self.feature_info = feature_info
        self.fpn_config = fpn_config

    def forward(self, x):

        nodes = []
        if len(self.inputs_offsets) == 0:
            return None
        for offset in self.inputs_offsets:
            in_features = x[offset]

            nodes.append(in_features)

        if self.weight_method == 'concat':
            # for node in nodes:
            #     print(node.shape)
            out = torch.cat(nodes, dim=1) # .permute(1, 0)[None, :, :]  # (1, C, M1 + M2 ...)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        return out


class FpnCombine(nn.Module):
    def __init__(self, feature_info, fpn_config, neck_cfg, fpn_channels, inputs_offsets,
                target_reduction, weight_method='attn'):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method
        self.feature_info = feature_info
        self.layer_cfg = neck_cfg.RESAMPLE_LAYER
        self.fpn_config = fpn_config

        self.resample = nn.ModuleDict()
        reduction_base = feature_info[0]['reduction']
        
        target_channels_idx = int(math.log(target_reduction // reduction_base, 2))
        for idx, offset in enumerate(inputs_offsets):
            if offset < len(feature_info):
                in_channels = feature_info[offset]['num_chs']
                input_reduction = feature_info[offset]['reduction']
            else:
                node_idx = offset
                input_reduction = fpn_config[node_idx]['reduction']
                # in_channels = fpn_config[node_idx]['num_chs']
                input_channels_idx = int(math.log(input_reduction // reduction_base, 2))
                in_channels = feature_info[input_channels_idx]['num_chs']

            reduction_ratio = target_reduction / input_reduction
            if weight_method == 'concat':
                self.resample[str(offset)] = ResampleFeatureMap(self.layer_cfg,
                    in_channels, in_channels, reduction_ratio=reduction_ratio)

        if weight_method == 'concat':
            src_channels = fpn_channels[target_channels_idx] * len(inputs_offsets)
            target_channels = fpn_channels[target_channels_idx]

    def forward(self, x, cur_xyz, cur_xyz_batch_cnt, global_roi_grid_points_list, batch_size):

        nodes = []
        if len(self.inputs_offsets) == 0:
            return None
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            in_features = x[offset]

            if offset < len(self.feature_info):
                # print(self.feature_info[offset]['reduction'])
                # print(global_roi_grid_points_list)
                xyz = global_roi_grid_points_list[self.feature_info[offset]['reduction']]
            else:
                xyz = global_roi_grid_points_list[self.fpn_config[offset]['reduction']]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(xyz.shape[1])

            new_xyz = cur_xyz.view(-1, 3)
            new_xyz_batch_cnt = cur_xyz_batch_cnt

            input_node = resample(xyz.view(-1, 3), xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, in_features)
            # print(in_features.shape)
            # print(offset)
            # print(input_node.shape)
            nodes.append(input_node)

        if self.weight_method == 'concat':
            # for node in nodes:
            #     print(node.shape)
            out = torch.cat(nodes, dim=1) # .permute(1, 0)[None, :, :]  # (1, C, M1 + M2 ...)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        return out


class Fnode(nn.Module):
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """
    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super(Fnode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine
        # self.after_combine = make_fc_layers(in_channels, out_channels, fc_list=[out_channels])

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, cur_xyz=None, cur_xyz_batch_cnt=None, global_roi_grid_points_list=None, batch_size=2) -> torch.Tensor:
        if cur_xyz is None:
            combine_feat = self.combine(x)
        else:
            combine_feat = self.combine(x, cur_xyz, cur_xyz_batch_cnt, global_roi_grid_points_list, batch_size)

        if combine_feat is None:
            return None
        else:
            combine_feat = self.after_combine(combine_feat.permute(1, 0)[None, :, :])
            combine_feat= combine_feat.squeeze(dim=0).permute(1, 0)
            return combine_feat


class PointGiraffeLayer(nn.Module):
    def __init__(self, feature_info, fpn_config, neck_cfg, inner_fpn_channels, outer_fpn_channels, num_levels=5, same_grid_for_all_levels=False):
        super(PointGiraffeLayer, self).__init__()
        self.num_levels = num_levels
        self.fpn_config = fpn_config
        self.same_grid_for_all_levels = same_grid_for_all_levels
        # self.conv_bn_relu_pattern = False

        self.feature_info = {}
        for idx, feat in enumerate(feature_info):
            self.feature_info[idx] = feat

        self.fnode = nn.ModuleList()
        reduction_base = feature_info[0]['reduction']
        for i, fnode_cfg in fpn_config.items():
            logging.debug('fnode {} : {}'.format(i, fnode_cfg))

            if fnode_cfg['is_out'] == 1:
                fpn_channels = outer_fpn_channels
            else:
                fpn_channels = inner_fpn_channels

            reduction = fnode_cfg['reduction']
            fpn_channels_idx = int(math.log(reduction // reduction_base, 2))
            if not same_grid_for_all_levels:
                combine = FpnCombine(
                    self.feature_info, fpn_config, neck_cfg, fpn_channels, tuple(fnode_cfg['inputs_offsets']),
                    target_reduction=reduction, weight_method=fnode_cfg['weight_method'])
            else:
                combine = FpnCombine_same_grid(
                    self.feature_info, fpn_config, neck_cfg, fpn_channels, tuple(fnode_cfg['inputs_offsets']),
                    target_reduction=reduction, weight_method=fnode_cfg['weight_method'])

            
            
            in_channels = 0
            out_channels = 0
            for input_offset in fnode_cfg['inputs_offsets']:
                in_channels += self.feature_info[input_offset]['num_chs']

            out_channels = fpn_channels[fpn_channels_idx]

            after_combine = nn.Sequential()
            after_combine.add_module('conv', make_fc_layers(in_channels, out_channels, fc_list=[out_channels]))
            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))
            # self.fnode.append(Fnode(combine, in_channels, out_channels))

            self.feature_info[i] = dict(num_chs=fpn_channels[fpn_channels_idx], reduction=reduction)

        self.out_feature_info = []
        out_node = list(self.feature_info.keys())[-num_levels::]
        for i in out_node:
            self.out_feature_info.append(self.feature_info[i])

        self.feature_info = self.out_feature_info
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                # print('INITIALISED')
                # print(m)
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)


    def forward(self, x: List[torch.Tensor], global_roi_grid_points_list=None, batch_size=2):
        for node_idx, fn in enumerate(self.fnode):
            if not self.same_grid_for_all_levels and (global_roi_grid_points_list is not None):
                # print('DEBUGGING {}'.format(node_idx+self.num_levels))
                cur_node_id = node_idx+self.num_levels
                cur_xyz = global_roi_grid_points_list[self.fpn_config[cur_node_id]['reduction']]
                cur_xyz_batch_cnt = cur_xyz.new_zeros(batch_size).int().fill_(cur_xyz.shape[1])
                x.append(fn(x, cur_xyz, cur_xyz_batch_cnt, global_roi_grid_points_list, batch_size))
            else:
                x.append(fn(x))

        return x[-self.num_levels::]