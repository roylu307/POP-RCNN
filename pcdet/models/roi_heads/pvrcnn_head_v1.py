import torch.nn as nn
import torch

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.roipoint_pool3d import roipoint_pool3d_utils
from ...utils import common_utils, box_utils
from .roi_head_template import RoIHeadTemplate


class PVRCNNHead_v1(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        self.raw_point_pool_layer, num_c_out_raw = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=1, config=self.model_cfg.ROI_GRID_POOL.POOL_LAYER.raw_points
        )

        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL.POOL_LAYER.point_features
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * (num_c_out + num_c_out_raw)

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )


        # from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
        # group0 = edict({'NUM_LOCAL_VOXEL': [ 2, 2, 2 ],
        #                'MAX_NEIGHBOR_DISTANCE': 0.2,
        #                'NEIGHBOR_NSAMPLE': -1,
        #                'POST_MLPS': [ 32, 32 ]})
        # group1 = edict({'NUM_LOCAL_VOXEL': [ 3, 3, 3 ],
        #                'MAX_NEIGHBOR_DISTANCE': 0.4,
        #                'NEIGHBOR_NSAMPLE': -1,
        #                'POST_MLPS': [ 32, 32 ]})
        # new_cfg = edict({'NAME': 'VectorPoolAggregationModuleMSG',
        #                 'NUM_GROUPS': 2,
        #                 'LOCAL_AGGREGATION_TYPE': 'local_interpolation',
        #                 'NUM_REDUCED_CHANNELS': 1,
        #                 'NUM_CHANNELS_OF_LOCAL_AGGREGATION': 32,
        #                 'MSG_POST_MLPS': [ 32 ],
        #                 'FILTER_NEIGHBOR_WITH_ROI': True,
        #                 'RADIUS_OF_NEIGHBOR_WITH_ROI': 2.4,
        #                 'GROUP_CFG_0': group0,
        #                 'GROUP_CFG_1': group1})

        # self.raw_point_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
        #     input_channels=1, config=self.model_cfg.ROI_GRID_POOL.raw_points
        # )



        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']
        raw_points = batch_dict['points']


        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)

        #######
        _, complement_pos_features = self.get_complement_pos_features(
            rois=rois, points=raw_points, new_xyz=global_roi_grid_points
        )
        # print(complement_pos_features)

        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)

        ###
        pooled_features = torch.cat([complement_pos_features, pooled_features], dim=-1)

        return pooled_features

    
    def get_complement_pos_features(self, rois, points, new_xyz):
        """
        Args:
            self
            rois: (B, num_rois, 7 + C)
            points: (num_rawpoints, 4) [bs_idx, x, y, z]
            new_xyz: (BxN, 6x6x6, 3)
        Returns:

        """

        batch_size = rois.shape[0]
        num_rois = rois.shape[1]

        batch_idx = points[:, 0]
        raw_xyz = points[:, 1:]

        sampled_points_list = []
        xyz_roi_cnt = []

        for bs_idx in range(batch_size):
            
            bs_mask = batch_idx == bs_idx

            sampled_points = raw_xyz[bs_mask][None, :, :] # (1, N, 3+C)
            sampled_rois = rois[bs_idx][None, :, :] # (1, num_rois(M), 7+C)
            
            import numpy as np
            # sample_extra_width = np.array([0.4])
            sample_extra_width = self.model_cfg.ROI_GRID_POOL.POOL_LAYER.raw_points.ROI_EXTRA_WIDTH
            sample_extra_width = sample_extra_width if len(sample_extra_width) == 3 else sample_extra_width[0] * np.array([1, 1, 1])

            enlarged_rois = box_utils.enlarge_box3d(sampled_rois.squeeze(dim=0),\
                                                    extra_width=sample_extra_width).unsqueeze(dim=0)
            
            point_assignment = roipoint_pool3d_utils.points_in_boxes_gpu(
                sampled_points[:, :, :3], enlarged_rois[:, :, 0:7]) # (1, N, M)
            point_mask = point_assignment.squeeze(0).permute(1,0)==1 # (M, N)
            num_point_in_rois_stack = point_mask.sum(dim=1)
            
            empty_flag = num_point_in_rois_stack == 0
            
            #### add point transformation here
            sampled_points, num_point_in_rois_stack = self.get_tranformed_points(
                sampled_points, enlarged_rois, point_mask, empty_flag, num_point_in_rois_stack)
            
            # sampled_points = torch.cat([(sampled_points.new_zeros(0, sampled_points.shape[-1]) if empty_flag[i] else sampled_points.squeeze(0)[point_mask[i,:],:]) \
            #                             for i in range(num_rois)], dim=0) # (R1 + R2 ..., 3)
            sampled_points_list.append(sampled_points)
            xyz_roi_cnt.append(num_point_in_rois_stack.int())

        batch_sampled_points = torch.cat(sampled_points_list, dim=0)
        batch_xyz_roi_cnt = torch.cat(xyz_roi_cnt, dim=0)
        new_xyz_roi_cnt = (new_xyz.new_ones(new_xyz.shape[0]) * new_xyz.shape[1]).int()

        pooled_points, pooled_features = self.raw_point_pool_layer(
            xyz=batch_sampled_points[:, :3].contiguous(),
            xyz_batch_cnt=batch_xyz_roi_cnt,
            new_xyz=new_xyz.view(-1, 3),
            new_xyz_batch_cnt=new_xyz_roi_cnt,
            features=batch_sampled_points[:, 3:].contiguous(),
        )

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)

        return pooled_points, pooled_features

    def get_tranformed_points(self, points, rois, point_mask, empty_flag, num_point_in_rois_stack):
        """
        Args:
            self
            points: (1, N, 4)
            rois: (1, num_rois, 7 + C)
            point_mask: (num_rois, N) 
            empty_flag: (num_rois)
            num_point_in_rois_stack: (num_rois) point count in rois
        Returns:

        """

        # points = points.view(-1, points.shape[-1])
        # rois = rois.view(-1, rois.shape[-1])

        all_points_list = []

        # if complement method
        num_rois = rois.shape[1]

        pt_transform_type = self.model_cfg.ROI_GRID_POOL.POOL_LAYER.raw_points.get('POINT_TRANSFORM_TYPE', None)
        if pt_transform_type == 'raw':
            return points, num_point_in_rois_stack

        for i in range(num_rois):
            if empty_flag[i]:
                # all_points = points.new_zeros(1, points.shape[-1])
                all_points = points[:, 0, :]
                num_point_in_rois_stack[i] = 1
            else:
                transformed_points = points[:, point_mask[i,:], 0:3]
                transformed_features = points[:, point_mask[i,:], 3:]
                
                if pt_transform_type != 'raw':

                    roi_center = rois[:, i, 0:3].clone()
                    transformed_points -= roi_center
                    
                    complement_points = common_utils.rotate_points_along_z(
                        transformed_points.clone(), -rois[:, i, 6])

                    if pt_transform_type == 'flip':
                        complement_points[:, :, 1] = -complement_points[:, :, 1]
                    elif pt_transform_type == 'rotate':
                        complement_points[:, :, :1] = -complement_points[:, :, :1]
                    elif pt_transform_type == 'raw1':
                        complement_points = complement_points
                    else:
                        raise NotImplementedError

                    complement_points += roi_center
                    complement_points = common_utils.rotate_points_along_z(
                        complement_points, rois[:, i, 6])

                    feat_transofrm_type = self.model_cfg.ROI_GRID_POOL.POOL_LAYER.raw_points.get('FEATURE_TRANSFORM_TYPE', 'equal')
                    if feat_transofrm_type == 'inverted':
                        complement_features = -transformed_features
                    elif feat_transofrm_type == 'equal':
                        complement_features = transformed_features
                    else:
                        raise NotImplementedError
                    
                    complement_points = torch.cat([complement_points, complement_features], dim=-1)
                    all_points = torch.cat([points[:, point_mask[i,:], :], complement_points], dim=1).squeeze(0)
                    num_point_in_rois_stack[i] = num_point_in_rois_stack[i]*2 # double point counts in rois
                
            all_points_list.append(all_points)

        all_points = torch.cat(all_points_list, dim=0)
        

        return all_points, num_point_in_rois_stack


    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
