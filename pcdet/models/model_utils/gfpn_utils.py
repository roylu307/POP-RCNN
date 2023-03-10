import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from collections import OrderedDict
from typing import List, Callable, Optional, Union, Tuple
import numpy as np
# from mmcv.cnn import build_norm_layer, build_conv_layer

# from timm import create_model
# from timm.models.layers import create_conv2d, create_pool2d, Swish, get_act_layer

# from ..utils import get_fpn_config, _init_weight_alt, _init_weight

_DEBUG = False

# _ACT_LAYER = Swish


class SequentialList(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""
    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x

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

# class ConvBnAct2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding='', bias=False,
#                  norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER):
#         super(ConvBnAct2d, self).__init__()

#         # ------ convolution --------------
#         dcn = dict(type='DCN', deform_groups=1)
#         padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
#         self.conv = build_conv_layer(None, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
#         # with dcn
#         # self.conv = build_conv_layer(dcn, in_channels, out_channels,kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
#         # ------- normalization ------------
#         norm_cfg = dict(type='SyncBN', requires_grad=True)
#         _, self.bn = build_norm_layer(norm_cfg, out_channels)
        
#         # ------- activation ---------------
#         self.act = None if act_layer is None else act_layer(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.act is not None:
#             x = self.act(x)
#         # return x



class Interpolate2d(nn.Module):
    r"""Resamples a 2d Image

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']
    name: str
    size: Optional[Union[int, Tuple[int, int]]]
    scale_factor: Optional[Union[float, Tuple[float, float]]]
    mode: str
    align_corners: Optional[bool]

    def __init__(self,
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
                 mode: str = 'nearest',
                 align_corners: bool = False) -> None:
        super(Interpolate2d, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == 'nearest' else align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners) #, recompute_scale_factor=False)


class ResampleFeatureMap(nn.Sequential):

    def __init__(
            self, in_channels, out_channels, reduction_ratio=1., pad_type='', downsample=None, upsample=None,
            norm_layer=nn.BatchNorm2d, apply_bn=False, conv_after_downsample=False, redundant_bias=False):
        super(ResampleFeatureMap, self).__init__()
        downsample = downsample or 'max'
        upsample = upsample or 'nearest'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.conv_after_downsample = conv_after_downsample

        conv = None
        if in_channels != out_channels:
            conv = make_fc_layer(in_channels, out_channels)

        if reduction_ratio > 1:
            if conv is not None and not self.conv_after_downsample:
                self.add_module('conv', conv)
            if downsample in ('max', 'avg'):
                stride_size = int(reduction_ratio) ** 2
                downsample = nn.MaxPool1d(kernel_size=stride_size, stride=stride_size)
                # downsample = create_pool2d(
                #      downsample, kernel_size=stride_size+1, stride=stride_size, padding=pad_type)
            else:
                downsample = Interpolate2d(scale_factor=1./reduction_ratio, mode=downsample)
            self.add_module('downsample', downsample)
            if conv is not None and self.conv_after_downsample:
                self.add_module('conv', conv)
        else:
            if conv is not None:
                self.add_module('conv', conv)
            if reduction_ratio < 1:
                scale = int(1 // reduction_ratio) ** 2
                self.add_module('upsample', Interpolate2d(scale_factor=scale, mode=upsample))

    # def forward(self, x):
    #     #  here for debugging only
    #     assert x.shape[1] == self.in_channels
    #     if self.reduction_ratio > 1:
    #         if hasattr(self, 'conv') and not self.conv_after_downsample:
    #             x = self.conv(x)
    #         x = self.downsample(x)
    #         if hasattr(self, 'conv') and self.conv_after_downsample:
    #             x = self.conv(x)
    #     else:
    #         if hasattr(self, 'conv'):
    #             x = self.conv(x)
    #         if self.reduction_ratio < 1:
    #             x = self.upsample(x)
    #     return x


class FpnCombine(nn.Module):
    def __init__(self, feature_info, fpn_config, fpn_channels, inputs_offsets, target_reduction, pad_type='',
                 downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, apply_resample_bn=False,
                 conv_after_downsample=False, redundant_bias=False, weight_method='attn'):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

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
                self.resample[str(offset)] = ResampleFeatureMap(
                    in_channels, in_channels, reduction_ratio=reduction_ratio, pad_type=pad_type,
                    downsample=downsample, upsample=upsample, norm_layer=norm_layer, apply_bn=apply_resample_bn,
                    conv_after_downsample=conv_after_downsample, redundant_bias=redundant_bias)
            else:
                self.resample[str(offset)] = ResampleFeatureMap(
                    in_channels, fpn_channels[target_channels_idx], reduction_ratio=reduction_ratio, pad_type=pad_type,
                    downsample=downsample, upsample=upsample, norm_layer=norm_layer, apply_bn=apply_resample_bn,
                    conv_after_downsample=conv_after_downsample, redundant_bias=redundant_bias)

        if weight_method == 'concat':
            src_channels = fpn_channels[target_channels_idx] * len(inputs_offsets)
            target_channels = fpn_channels[target_channels_idx]

        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        dtype = x[0].dtype
        nodes = []
        if len(self.inputs_offsets) == 0:
            return None
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
            out = torch.sum(out, dim=-1)
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)
            out = torch.sum(out, dim=-1)
        elif self.weight_method == 'sum':
            out = torch.stack(nodes, dim=-1)
            out = torch.sum(out, dim=-1)
        elif self.weight_method == 'concat':
            # for node in nodes:
                # print(node.shape)
            out = torch.cat(nodes, dim=1)
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

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        combine_feat = self.combine(x)
        if combine_feat is None:
            return None
        else:
            return self.after_combine(combine_feat)


class GiraffeLayer(nn.Module):
    def __init__(self, feature_info, fpn_config, inner_fpn_channels, outer_fpn_channels, num_levels=5, pad_type='',
                 downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, act_layer=None,
                 apply_resample_bn=False, conv_after_downsample=True, conv_bn_relu_pattern=False,
                 separable_conv=True, redundant_bias=False):
        super(GiraffeLayer, self).__init__()
        self.num_levels = num_levels
        self.conv_bn_relu_pattern = False

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
            combine = FpnCombine(
                self.feature_info, fpn_config, fpn_channels, tuple(fnode_cfg['inputs_offsets']),
                target_reduction=reduction, pad_type=pad_type, downsample=downsample, upsample=upsample,
                norm_layer=norm_layer, apply_resample_bn=apply_resample_bn, conv_after_downsample=conv_after_downsample,
                redundant_bias=redundant_bias, weight_method=fnode_cfg['weight_method'])

            after_combine = nn.Sequential()
            
            in_channels = 0
            out_channels = 0
            for input_offset in fnode_cfg['inputs_offsets']:
                in_channels += self.feature_info[input_offset]['num_chs']

            out_channels = fpn_channels[fpn_channels_idx]
            
            # after_combine.add_module('conv1x1', create_conv2d(in_channels, out_channels, kernel_size=1))
            # conv_kwargs = dict(
            #     in_channels=out_channels, 
            #     out_channels=out_channels, 
            #     kernel_size=3, padding=pad_type,
            #     bias=False, norm_layer=norm_layer, act_layer=act_layer)
            # if not conv_bn_relu_pattern:
            #     conv_kwargs['bias'] = redundant_bias
            #     conv_kwargs['act_layer'] = None
            #     after_combine.add_module('act', act_layer(inplace=True))
            # after_combine.add_module(
            #     'conv', SeparableConv2d(**conv_kwargs) if separable_conv else ConvBnAct2d(**conv_kwargs))

            after_combine.add_module('conv', make_fc_layers(in_channels, out_channels, fc_list=[out_channels]))

            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))
            self.feature_info[i] = dict(num_chs=fpn_channels[fpn_channels_idx], reduction=reduction)

        self.out_feature_info = []
        out_node = list(self.feature_info.keys())[-num_levels::]
        for i in out_node:
            self.out_feature_info.append(self.feature_info[i])

        self.feature_info = self.out_feature_info

    def forward(self, x: List[torch.Tensor]):
        for node_idx, fn in enumerate(self.fnode):
            # print('DEBUGGING {}'.format(node_idx+4))
            x.append(fn(x))
        return x[-self.num_levels::]