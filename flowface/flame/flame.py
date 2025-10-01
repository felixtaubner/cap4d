"""
This code is adapted from the Metrical Photometric Tracker repository:
https://github.com/Zielon/metrical-tracker/tree/master

Original implementation by the Max Planck Institute for Intelligent Systems.

Modifications:
- Optimized computations
- Convert pytorch3d coordinates into OpenCV coordinates
- Added blink blendshape

License: Please refer to the LICENSE file in the MPT repository for the terms
of use and redistribution: https://github.com/Zielon/metrical-tracker/blob/master/LICENSE
"""

from typing import Dict, Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F

from flowface.flame.io import load_model_pkl
from flowface.flame.utils import (
    batch_rodrigues, 
    OPENCV2PYTORCH3D, 
    transform_vertices, 
    project_vertices
)


FLAME_N_SHAPE = 300
FLAME_N_EXPR = 100
FLAME_N_VERTS = 5023


# This module has NO trainable parameters
class FlameSkinner(torch.nn.Module):
    """
    Module which generates skinned FLAME meshes given the shape, expression and pose parameters.
    There are no trainable parameters -- most members are essentially frozen weights of FLAME.
    The forward() function can optionally use pre-annotated 3D landmarks on a FLAME template mesh
     to generate skinned 3D landmarks.
    Rotation limits can optionally be applied to the joint rotations to keep them within certain limits.
    """

    template_vertices: torch.Tensor  # float32 (5023, 3)
    template_faces: torch.Tensor  # float32 (5023, 3)
    shape_eigenvecs: torch.Tensor
    expr_eigenvecs: torch.Tensor
    pose_eigenvecs: torch.Tensor  # float32 (36, 5023*3)
    joint_regressor: torch.Tensor  # float32 (5, 5023)
    joint_parents: torch.Tensor  # int64 (5)
    skinning_weights: torch.Tensor  # float32 (5023, 5)

    def __init__(
        self,
        flame_pkl_path: str,
        n_shape_params: int = FLAME_N_SHAPE,
        n_expr_params: int = FLAME_N_EXPR,
        blink_blendshape_path: str = None,
    ):
        """
        Parameters
        ----------
            flame_pkl_path: str
                Path to pkl file containing the flame blendshapes and parameters
            n_shape_params: int
                Number of shape parameters to use up to a max of 300
            n_expr_params: int
                Number of expression parameters to use up to a max of 100
            blink_blendshape_path: str
                Path to npy file containing the blink blendshapes - if None, no blink blendshape will be used
        """

        super().__init__()

        assert n_shape_params <= FLAME_N_SHAPE
        assert n_expr_params <= FLAME_N_EXPR
        
        # load FLAME weight pickle file with monkey-patch
        flame_dict = load_model_pkl(flame_pkl_path)

        # initialize vertex position components
        shape_eigenvecs = torch.tensor(
            flame_dict["shapedirs"][..., :n_shape_params]
        )
        expr_eigenvecs = torch.tensor(
            flame_dict["shapedirs"][..., FLAME_N_SHAPE:FLAME_N_SHAPE+n_expr_params]
        )

        if blink_blendshape_path is not None:
            blink_blendshape = torch.tensor(np.load(blink_blendshape_path))
            expr_eigenvecs[:, :, -1] = blink_blendshape

        template_vertices = torch.tensor(flame_dict["v_template"])
        template_faces = torch.tensor(flame_dict["f"]).long()
        pose_eigenvecs = torch.tensor(flame_dict["posedirs"])
        pose_eigenvecs = einops.rearrange(
            pose_eigenvecs, "v xyz j -> j (v xyz)"
        )
        joint_regressor = torch.tensor(flame_dict["J_regressor"])
        joint_parents = torch.tensor(flame_dict["kintree_table"][0]).long()
        skinning_weights = torch.tensor(flame_dict["weights"])

        self.n_shape_params = n_shape_params
        self.n_expr_params = n_expr_params

        # register buffers
        self.register_buffer("template_vertices", template_vertices, persistent=False)
        self.register_buffer("template_faces", template_faces, persistent=False)
        self.register_buffer("shape_eigenvecs", shape_eigenvecs, persistent=False)
        self.register_buffer("expr_eigenvecs", expr_eigenvecs, persistent=False)
        self.register_buffer("pose_eigenvecs", pose_eigenvecs, persistent=False)
        self.register_buffer("joint_regressor", joint_regressor, persistent=False)
        self.register_buffer("joint_parents", joint_parents, persistent=False)
        self.register_buffer("skinning_weights", skinning_weights, persistent=False)

        self.cached_shape_eigenvecs = None
        self.cached_expr_eigenvecs = None
        self.cached_j_regressor = None
        self.cached_lbs_weights = None
        self.cached_pose_dirs = None
        self.cached_template_vertices = None

    def _get_template_vertices(
        self,
        vert_mask: torch.Tensor = None,
    ):
        if vert_mask is not None:
            if self.cached_template_vertices is None:
                self.cached_template_vertices = self.template_vertices[None, vert_mask]
            return self.cached_template_vertices
        
        return self.template_vertices[None]

    def _get_shape_offsets(
        self, 
        shape_params: torch.Tensor, # float32 (B, n_shape_params)
        vert_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        assert shape_params.shape[1] == self.n_shape_params

        if vert_mask is not None:
            if self.cached_shape_eigenvecs is None:
                self.cached_shape_eigenvecs = self.shape_eigenvecs[vert_mask]
            shape_eigenvecs = self.cached_shape_eigenvecs
        else:
            shape_eigenvecs = self.shape_eigenvecs

        return einops.einsum(
            shape_params, 
            shape_eigenvecs,
            "b betas, V xyz betas -> b V xyz"
        )
    
    def _get_expr_offsets(
        self,
        expr_params: torch.Tensor, # float32 (B, n_expr_params)
        vert_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        assert expr_params.shape[1] == self.n_expr_params

        if vert_mask is not None:
            if self.cached_expr_eigenvecs is None:
                self.cached_expr_eigenvecs = self.expr_eigenvecs[vert_mask]
            expr_eigenvecs = self.cached_expr_eigenvecs
        else:
            expr_eigenvecs = self.expr_eigenvecs

        return einops.einsum(
            expr_params, 
            expr_eigenvecs,
            "b betas, V xyz betas -> b V xyz"
        )
    
    def _apply_joint_rotation(
        self,
        vertices: torch.Tensor, # float32 (B, 5023, 3)
        rotations: torch.Tensor, # joint rotation matrices (B, J, 3, 3) [base, neck, jaw, eye1, eye2]
        return_joints: bool = False, # whether or not to return joint positions
        return_transforms: bool = False, # whether or not to return per vertex transforms
        vert_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        
        j_regressor = self.joint_regressor
        lbs_weights = self.skinning_weights
        pose_dirs = einops.rearrange(
            self.pose_eigenvecs, "(J i j) (V xyz) -> J i j V xyz", i=3, j=3, xyz=3
        )  # (4, 3, 3, 5023, 3)

        if vert_mask is not None:
            if self.cached_j_regressor is None:
                self.cached_j_regressor = j_regressor[:, vert_mask]
                self.cached_lbs_weights = lbs_weights[vert_mask]
                self.cached_pose_dirs = pose_dirs[:, :, :, vert_mask]

            j_regressor = self.cached_j_regressor
            lbs_weights = self.cached_lbs_weights
            pose_dirs = self.cached_pose_dirs

        identity = torch.eye(3, dtype=torch.float32, device=vertices.device)  # (3, 3)
        # pose_offsets = einops.einsum(rotations[:, 1:] - identity, pose_dirs, "B J i j, J i j V xyz -> B V xyz")
        # WARNING: This is because we reverse neck rotation definition. We rotate neck using base rotation param.
        pose_offset_params = rotations[:, [0, 2, 3, 4]] - identity
        # import pdb; pdb.set_trace()
        # pose_offset_params = torch.cat([
        #     -rotations[:, [1]],
        #     rotations[:, [2, 3, 4]],
        # ], dim=1)
        pose_offset_params = pose_offset_params - identity
        pose_offsets = einops.einsum(pose_offset_params, pose_dirs, "B J i j, J i j V xyz -> B V xyz")

        assert rotations.shape[1] == j_regressor.shape[0]

        # Get joint locations
        joints = einops.einsum(vertices, j_regressor, "b V xyz, J V -> b J xyz") # (B, J, 3)

        v_posed = vertices + pose_offsets

        transforms = F.pad(rotations, [0, 1, 0, 1, 0, 0, 0, 0])
        transforms[..., -1, -1] = 1.
        transforms[..., :3, -1] = joints - (rotations @ joints[..., None])[..., 0]
        weighted_transforms = einops.einsum(
            lbs_weights, transforms, "V J, b J i j -> b V i j"
        )  # (B, 5023, 3, 3)
        v_posed_homo = F.pad(v_posed, [0, 1], value=1)  # (B, 5023, 4)
        v_rotated = einops.einsum(
            weighted_transforms, v_posed_homo, "b V i j, b V j -> b V i"
        )  # (B, 5023, 4)

        output = [v_rotated[..., :3]]

        if return_joints:
            output.append(joints)
        if return_transforms:
            output.append(weighted_transforms)
            
        return output
        
    def _get_gaze_dirs(
        self,
        base_rot: torch.Tensor, # base rotation matrix (B, 3, 3)
        rotations: torch.Tensor, # joint rotation matrices (B, J, 3, 3) [base, neck, jaw, eye1, eye2]
    ):
        forward = torch.tensor([0., 0., 1.], device=rotations.device)
        gaze_dirs = base_rot[:, None] @ rotations[:, 3:5] @ forward[None, None, :, None]  # (B, 2, 3, 3) @ (1, 1, 3, 1)

        return gaze_dirs[..., 0]  # (B, 2, 3, 1)
        
    def forward(
        self,
        flame_sequence: Dict,
        vert_mask: torch.Tensor = None,
        return_gaze: bool = False,
    ) -> torch.Tensor:
        """
        Compute 3D vertices given a flame sequence with shape parameters
        flame_sequence (dictionary):
            shape (N_shape)
            and N_t timesteps of expression, pose (rot, tra), eye_rot (optional), jaw_rot (optional):
            expr (N_t, N_exp)
            rot (N_t, 3)
            tra (N_t, 3)
            eye_rot (N_t, 3)
            jaw_rot (N_t, 3)
        vert_mask: torch.Tensor (N_v)
            If given, will apply a vertex mask to save computation time.

        output: verts (N_t, V, 3)
        """

        shape_offsets = self._get_shape_offsets(flame_sequence["shape"][None], vert_mask)
        shape_verts = self._get_template_vertices(vert_mask) + shape_offsets

        expr_offsets = self._get_expr_offsets(flame_sequence["expr"], vert_mask)

        verts = shape_verts + expr_offsets  # N_t, V, 3

        # create rotation matrix for joint rotations, we apply base transform separately
        rotations = torch.eye(3, device=verts.device)[None, None].repeat(verts.shape[0], 5, 1, 1)
        if "neck_rot" in flame_sequence and flame_sequence["neck_rot"] is not None:
            rotations[:, 1, ...] = batch_rodrigues(flame_sequence["neck_rot"])
        if "jaw_rot" in flame_sequence and flame_sequence["jaw_rot"] is not None:
            rotations[:, 2, ...] = batch_rodrigues(flame_sequence["jaw_rot"])
        if "eye_rot" in flame_sequence and flame_sequence["eye_rot"] is not None:
            eye_rot = batch_rodrigues(flame_sequence["eye_rot"])
            rotations[:, 3, ...] = eye_rot
            rotations[:, 4, ...] = eye_rot

        verts = self._apply_joint_rotation(verts, rotations=rotations, vert_mask=vert_mask)[0]

        # apply base transform separately
        base_rot = batch_rodrigues(flame_sequence["rot"])
        base_tra = flame_sequence["tra"][..., None]
        verts = (base_rot @ verts.permute(0, 2, 1) + base_tra).permute(0, 2, 1)

        if return_gaze:
            gaze_dirs = self._get_gaze_dirs(base_rot, rotations)
            return verts, gaze_dirs
        else:
            return verts


def compute_flame(
    flame: FlameSkinner, 
    fit_3d: Dict,
):
    flame_sequence = {
        "shape": torch.tensor(fit_3d["shape"]).float(),
        "expr": torch.tensor(fit_3d["expr"]).float(),
        "rot": torch.tensor(fit_3d["rot"]).float(),
        "tra": torch.tensor(fit_3d["tra"]).float(),
        "eye_rot": torch.tensor(fit_3d["eye_rot"]).float(),
        "jaw_rot": None, 
    }
    if "jaw_rot" in fit_3d:
        flame_sequence["jaw_rot"] = torch.tensor(fit_3d["jaw_rot"]).float()

    # compute FLAME vertices
    verts_3d = flame(
        flame_sequence, 
        vert_mask=None, 
    )  # [N_t V 3], [N_t 2 3]

    fx, fy, cx, cy = [torch.tensor(fit_3d[key]).float() for key in ["fx", "fy", "cx", "cy"]]  # [N_C, 1]
    extr = torch.tensor(fit_3d["extr"]).float()  # [N_c 3 3]
    cam_parameters = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "extr": extr,
    }

    # transform into OpenCV camera coordinate convention
    verts_3d_cv = transform_vertices(OPENCV2PYTORCH3D[None].to(verts_3d.device), verts_3d)  # [N_t V 3]
    # project vertices to cameras
    verts_2d = project_vertices(verts_3d_cv, cam_parameters)  # [N_c N_t V 3]

    return {
        "verts_3d": verts_3d.cpu().numpy(),
        "verts_3d_cv": verts_3d_cv.cpu().numpy(),
        "verts_2d": verts_2d.cpu().numpy(),
    }
