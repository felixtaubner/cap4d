from typing import Dict, Optional

import einops
import numpy as np
import torch

from flowface.flame.flame import FlameSkinner, FLAME_N_SHAPE, FLAME_N_EXPR
from flowface.flame.utils import batch_rodrigues, OPENCV2PYTORCH3D, transform_vertices, project_vertices

from cap4d.flame.mouth import FlameMouth


FLAME_PKL_PATH = "data/assets/flame/flame2023_no_jaw.pkl"
JAW_REGRESSOR_PATH = "data/assets/flame/jaw_regressor.npy"
BLINK_BLENDSHAPE_PATH = "data/assets/flame/blink_blendshape.npy"


# This module has NO trainable parameters
class CAP4DFlameSkinner(FlameSkinner):
    def __init__(
        self,
        flame_pkl_path: str  = FLAME_PKL_PATH,
        n_shape_params: int = FLAME_N_SHAPE,
        n_expr_params: int = FLAME_N_EXPR,
        blink_blendshape_path: str = BLINK_BLENDSHAPE_PATH,
        add_mouth: bool = False,
        add_lower_jaw: bool = False,
        jaw_regressor_path: str = JAW_REGRESSOR_PATH,
    ):
        super().__init__(flame_pkl_path, n_shape_params, n_expr_params, blink_blendshape_path)

        self.add_mouth = add_mouth
        if add_mouth:
            self.mouth = FlameMouth()
        
        self.add_lower_jaw = add_lower_jaw
        if add_lower_jaw:
            self.lower_jaw = FlameMouth()
            jaw_regressor = torch.tensor(np.load(jaw_regressor_path))
            self.register_buffer("jaw_regressor", jaw_regressor)
    
    def forward(
        self,
        flame_sequence: Dict,
        return_offsets: bool = True,
        return_transforms: bool = False,
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
            neck_rot (N_t, 3)

        output: verts (N_t, V, 3)
        """
        shape_offsets = self._get_shape_offsets(flame_sequence["shape"][None], None)
        shape_verts = self._get_template_vertices(None) + shape_offsets

        expr_offsets = self._get_expr_offsets(flame_sequence["expr"], None)

        verts = shape_verts + expr_offsets  # N_t, V, 3

        # create rotation matrix for joint rotations, we apply base transform separately
        rotations = torch.eye(3, device=verts.device)[None, None].repeat(verts.shape[0], 5, 1, 1)
        if "neck_rot" in flame_sequence and flame_sequence["neck_rot"] is not None:
            rotations[:, 0, ...] = batch_rodrigues(flame_sequence["neck_rot"])
        if "jaw_rot" in flame_sequence and flame_sequence["jaw_rot"] is not None:
            rotations[:, 2, ...] = batch_rodrigues(flame_sequence["jaw_rot"])
        if "eye_rot" in flame_sequence and flame_sequence["eye_rot"] is not None:
            eye_rot = batch_rodrigues(flame_sequence["eye_rot"])
            rotations[:, 3, ...] = eye_rot
            rotations[:, 4, ...] = eye_rot

        verts, v_transforms = self._apply_joint_rotation(verts, rotations=rotations, vert_mask=None, return_transforms=True)

        # compute offsets (including joint rotations)
        offsets = verts - shape_verts
        if self.add_mouth:
            mouth_verts = self.mouth(shape_verts, self.joint_regressor)
            mouth_verts = mouth_verts.repeat(verts.shape[0], 1, 1)
            verts = torch.cat([verts, mouth_verts], dim=1)
            offsets = torch.cat([offsets, torch.zeros_like(mouth_verts)], dim=1)
            v_transforms = torch.cat([v_transforms, torch.zeros(mouth_verts.shape[0], mouth_verts.shape[1], 4, 4, device=v_transforms.device)], dim=1)
        if self.add_lower_jaw:
            jaw_rot = einops.einsum(flame_sequence["expr"], self.jaw_regressor, 'b exp, exp r -> b r')
            neutral_jaw_verts = self.lower_jaw(shape_verts, self.joint_regressor, batch_rodrigues(jaw_rot * 0.))
            jaw_verts = self.lower_jaw(shape_verts, self.joint_regressor, batch_rodrigues(jaw_rot))
            verts = torch.cat([verts, jaw_verts], dim=1)
            offsets = torch.cat([offsets, jaw_verts - neutral_jaw_verts], dim=1)
            jaw_transforms = torch.zeros(jaw_verts.shape[0], 4, 4, device=v_transforms.device)
            jaw_transforms[:, :3, :3] = batch_rodrigues(jaw_rot)
            jaw_transforms[..., -1, -1] = 1.
            jaw_transforms = jaw_transforms[:, None].repeat(1, jaw_verts.shape[1], 1, 1)
            v_transforms = torch.cat([v_transforms, jaw_transforms], dim=1)

        # apply base transform separately
        base_rot = batch_rodrigues(flame_sequence["rot"])
        base_tra = flame_sequence["tra"][..., None]
        verts = (base_rot @ verts.permute(0, 2, 1) + base_tra).permute(0, 2, 1)

        output = [verts]

        if return_offsets:
            output.append(offsets)
        if return_transforms:
            base_transform = torch.cat([base_rot, base_tra], dim=2)
            base_transform = torch.cat([base_transform, torch.zeros_like(base_transform[:, :1, ...])], dim=1)
            base_transform[..., -1, -1] = 1.
            v_transforms = einops.einsum(base_transform, v_transforms, 'b i j, b N j k -> b N i k')

            output.append(v_transforms)

        return output


def compute_flame(
    flame: CAP4DFlameSkinner, 
    fit_3d: Dict[str, np.ndarray],
):
    flame_sequence = {
        "shape": torch.tensor(fit_3d["shape"]).float(),
        "expr": torch.tensor(fit_3d["expr"]).float(),
        "rot": torch.tensor(fit_3d["rot"]).float(),
        "tra": torch.tensor(fit_3d["tra"]).float(),
        "eye_rot": torch.tensor(fit_3d["eye_rot"]).float(),
        "jaw_rot": None, 
        "neck_rot": None, 
    }
    if "neck_rot" in fit_3d:
        flame_sequence["neck_rot"] = torch.tensor(fit_3d["neck_rot"]).float()
    if "jaw_rot" in fit_3d:
        flame_sequence["jaw_rot"] = torch.tensor(fit_3d["jaw_rot"]).float()

    # compute FLAME vertices
    verts_3d, offsets_3d = flame(
        flame_sequence, 
        return_offsets=True,
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
        "offsets_3d": offsets_3d.cpu().numpy(),
    }
