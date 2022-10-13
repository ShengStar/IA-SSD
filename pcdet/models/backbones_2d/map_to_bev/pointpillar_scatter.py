import torch
import torch.nn as nn
from ....ops.pointnet2.pointnet2_batch import pointnet2_utils
tv = None
try:
    import cumm.tensorview as tv
except:
    pass

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

        self.ctr_idx_list = [-1, -1, -1, -1, -1, 5]
        self.layer_inputs = [0, 1, 2, 3, 4, 3]
        self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=[0.16, 0.16, 4],
                coors_range_xyz=[0, -39.68, -3, 69.12, 39.68, 1],
                num_point_features=4,
                max_num_points_per_voxel=32,
                max_num_voxels=16000,
            )
    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict, **kwargs):
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None ###

        encoder_xyz, encoder_features, sa_ins_preds = [xyz], [features], []
        encoder_coords = [torch.cat([batch_idx.view(batch_size, -1, 1), xyz], dim=-1)]

        li_cls_pred = None
        xyz_input = encoder_xyz[self.layer_inputs[0]] # [8, 16384, 3]
        feature_input = encoder_features[self.layer_inputs[0]] # [8, 1, 16384]

        ctr_xyz = encoder_xyz[self.ctr_idx_list[0]] if self.ctr_idx_list[0] != -1 else None
        # li_xyz, li_features, li_cls_pred = self.SA_modules[i](xyz_input, feature_input, li_cls_pred, ctr_xyz=ctr_xyz)
        npoint = 4096
        xyz_flipped = xyz.transpose(1, 2).contiguous() 
        sample_idx = pointnet2_utils.furthest_point_sample(xyz.contiguous(), npoint)
        new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sample_idx).transpose(1, 2).contiguous()
        new_features = pointnet2_utils.gather_operation(features, sample_idx).transpose(1, 2).contiguous()
        new_points = torch.cat([new_xyz, new_features], 2)
        voxels_reduce = []
        coordinates_reduce = []
        num_points_reduce = []

        for i in range(batch_size):
            points = new_points[i,:]
            voxel_output = self.voxel_generator.generate(points.cpu().numpy())
            voxels_re, coordinates_re, num_points_re = voxel_output
            voxels_reduce.append(voxels_re)
            coordinates_reduce.append(coordinates_re)
            num_points_reduce.append(num_points_re)


        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            coor = coordinates_reduce[batch_idx]
            coor_indices = coor[:, 0] + coor[:, 1] * self.nx + coor[:, 2]
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            spatial_feature_1 = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            spatial_feature_1[:, coor_indices] = spatial_feature[:,coor_indices]
            batch_spatial_features.append(spatial_feature_1)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict
