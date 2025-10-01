from typing import Dict, List, Tuple
from pathlib import Path
import argparse
import shutil
import json

import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import roma
import trimesh
from decord import VideoReader
from scipy.ndimage import gaussian_filter1d

from flowface.flame.utils import (
    OPENCV2PYTORCH3D, 
    transform_vertices, 
    batch_rodrigues, 
    project_vertices,
)
from cap4d.flame.flame import CAP4DFlameSkinner
from cap4d.datasets.utils import FrameReader, pivot_camera_intrinsic

from scripts.pixel3dmm.l2cs_eye_tracker import L2CSTracker, compute_eyeball_rotation
from scripts.pixel3dmm.robust_video_matting.model.model import MattingNetwork


ORBIT_PERIOD = 8  # orbit period in seconds
ORBIT_AMPLITUDE_YAW = 55  # yaw angle amplitude for orbit
ORBIT_AMPLITUDE_PITCH = 20  # pitch angle amplitude for orbit
MAX_EYE_ROTATION = 25  # maximum eyeball rotation angle in degrees


class FlameFittingModel(nn.Module):
    def __init__(
        self,
        flame: CAP4DFlameSkinner,
        n_timesteps: int,
        vertex_weights: torch.Tensor,
        use_jaw_rotation: bool = False,
    ):
        """
        flame: the flame model to use (Note: vertex mask needs to be applied to this model)
        cam_resolutions: resolutions of the cameras
        n_timesteps: number of timesteps
        n_cams: number of cameras
        fps: frames per second of the sequence for acceleration loss calculation - if fps < 0, no acceleration loss
        use_jaw_rotation: whether or not to use jaw rotation (make sure the appropriate flame model is loaded)
        camera_calibration: camera calibration dictionary containing
            "intrinsics": torch.Tensor with (N_c, 3, 3) intrinsic matrices (K)
            "extrinsics": torch.Tensor with (N_c, 4, 4) extrinsic transforms (RT)
        vertex_mask: torch.Tensor with (N_v) indicating the vertices used for tracking
        """
        super().__init__()

        n_expr_params = flame.n_expr_params
        n_shape_params = flame.n_shape_params
        self.use_jaw_rotation = use_jaw_rotation
        self.n_timesteps = n_timesteps

        self.flame = flame

        # initialize flame sequence parameters
        self.shape = nn.Parameter(torch.zeros(n_shape_params))
        self.expr = nn.Parameter(torch.zeros(n_timesteps, n_expr_params))
        self.rot = nn.Parameter(torch.zeros(n_timesteps, 3))
        self.tra = nn.Parameter(torch.zeros(n_timesteps, 3))
        self.eye_rot = nn.Parameter(torch.zeros(n_timesteps, 3))
        self.neck_rot = nn.Parameter(torch.zeros(n_timesteps, 3))
        if use_jaw_rotation:
            self.jaw_rot = nn.Parameter(torch.zeros(n_timesteps, 3))
            # some heuristic normal deviations of jaw rotations for loss calculation
            self.register_buffer("jaw_std", torch.deg2rad(torch.tensor([45, 5, 0.01])), persistent=False)
        else:
            self.jaw_rot = None
        
        # utils
        # self.register_buffer("vertex_mask", vertex_mask, persistent=False)
        self.register_buffer("vertex_weights", vertex_weights / vertex_weights.sum(), persistent=False)
        self.register_buffer("opencv2pytorch", OPENCV2PYTORCH3D, persistent=False)
    
    def _compute_reg_losses(self, verts_3d: torch.Tensor):
        l_shape = (self.shape ** 2).sum(dim=-1).mean()

        expr_params = self.expr
        if self.use_jaw_rotation:
            # normalize jaw rotation values
            jaw_values = self.jaw_rot / self.jaw_std[None]
            expr_params = torch.cat([expr_params, jaw_values], dim=-1)

        l_expr = (expr_params ** 2).sum(dim=-1).mean()

        return {
            "l_shape": l_shape,
            "l_expr": l_expr,
        }

    def forward(self):
        flame_sequence = {
            "shape": self.shape,
            "expr": self.expr,
            "rot": self.rot,
            "tra": self.tra,
            "eye_rot": self.eye_rot,
            "jaw_rot": self.jaw_rot,
            "neck_rot": self.neck_rot,
        }

        # compute FLAME vertices
        verts_3d, _ = self.flame(
            flame_sequence, 
        )

        # transform into OpenCV camera coordinate convention
        verts_3d_cv = transform_vertices(self.opencv2pytorch[None], verts_3d)  # [N_t V 3]

        return {
            "verts_3d": verts_3d_cv,
        }

    def fit(
        self,
        verts_3d: torch.Tensor,
        init_lr: float = 1e-2,
        n_steps: int = 6000,
        w_shape_reg: float = 1e-2,
        w_expr_reg: float = 1e-2,
        verbose: bool = True,
        pos_warm_up_steps: int = 500,
    ):
        """
        Fit 3D FLAME model to 2D alignment computed by the alignment module
        init_lr: initial learning rate
        n_steps: number of fitting steps
        w_shape_reg: FLAME shape parameter regularization
        w_expr_reg: FLAME expression parameter regularization
        """

        opt = torch.optim.Adam(
            lr=init_lr, params=self.parameters(), betas=(0.96, 0.999), 
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=100, factor=0.5
        )

        if verbose:
            pbar = tqdm(range(n_steps))
        else:
            pbar = range(n_steps)
        for i in pbar:
            if i < pos_warm_up_steps:
                self.expr.data *= 0.
                self.shape.data *= 0.
                self.eye_rot.data *= 0.
            # self.neck_rot.data = torch.zeros_like(self.neck_rot.data)  # * 0.5

            opt.zero_grad(set_to_none=True)

            output_dict = self.forward()

            # compute 2D alignment loss
            l_vert = (output_dict["verts_3d"] - verts_3d) / 0.01  # [N_c N_t V 2]
            l_vert = l_vert.norm(dim=-1)
            l_vert_max = l_vert.max()
            l_vert = l_vert ** 2 ## SQURE HACK
            l_vert = (l_vert * self.vertex_weights[None]).sum(dim=-1)
            l_vert = l_vert.mean()  # apply valid view mask

            reg_dict = self._compute_reg_losses(output_dict["verts_3d"])

            loss = l_vert
            loss += reg_dict["l_shape"] * w_shape_reg
            loss += reg_dict["l_expr"] * w_expr_reg

            loss = loss.mean()
            loss.backward()

            opt.step()

            if i > pos_warm_up_steps:
                scheduler.step(loss.item())

            if opt.param_groups[0]["lr"] < 1e-5:
                break

            if i % 10 == 0 and verbose:
                desc = ""
                desc += f"lr: {opt.param_groups[0]['lr']}, "
                desc += f"l: {loss.item():.3f}, "
                desc += f"vert: {l_vert.item():.3f}, "
                desc += f"vert_max: {l_vert_max.item():.3f}, "
                desc += f"shp: {reg_dict['l_shape'].item():.2f}, "
                desc += f"expr: {reg_dict['l_expr'].item():.2f}, "
                desc += f"shape_max: {self.shape.max().item():.2f}, "
                desc += f"expr_max: {self.expr.max().item():.2f}, "
                pbar.set_description(desc)

        return l_vert, output_dict["verts_3d"]

    def export_results(self):
        """
        Return the fitted parameters.
        """

        fit_3d = {
            "shape": self.shape.data.detach().cpu().numpy(),
            "expr": self.expr.data.detach().cpu().numpy(),
            "rot": self.rot.data.detach().cpu().numpy(),
            "tra": self.tra.data.detach().cpu().numpy(),
            "eye_rot": self.eye_rot.data.detach().cpu().numpy(),
            "neck_rot": self.neck_rot.data.detach().cpu().numpy(),
        }
        if self.jaw_rot is not None:
            fit_3d["jaw_rot"] = self.jaw_rot.data.detach().cpu().numpy()
        
        return fit_3d


def fit_flame(
    verts_3d, 
    gaze_directions,
    cam_rt,
    use_jaw_rotation=False,
    n_shape_params=150, 
    n_expr_params=65, 
    device="cpu",
    n_steps: int = 6000,
    w_shape_reg: float = 1e-2,
    w_expr_reg: float = 1e-2,
    smooth_eye_rotations: bool = False,
):
    verts_3d = torch.tensor(verts_3d).float().to(device)

    # vert_mask = torch.tensor(np.load("data/assets/flame/flowface_vertex_mask.npy"))
    vert_weights = torch.tensor(np.load("data/assets/flame/flowface_vertex_weights.npy"))
    # vert_weights = torch.ones_like(vert_weights)
    # vert_weights = vert_weights * vert_mask
    # import pdb; pdb.set_trace()

    flame_path = "data/assets/flame/flame2023_no_jaw.pkl"
    if use_jaw_rotation:
        flame_path = "data/assets/flame/flame2023.pkl"

    flame = CAP4DFlameSkinner(
        flame_path, 
        n_shape_params=n_shape_params, 
        n_expr_params=n_expr_params, 
        blink_blendshape_path="data/assets/flame/blink_blendshape.npy",
    ).to(device)

    fitter = FlameFittingModel(
        flame, 
        verts_3d.shape[0], 
        vertex_weights=vert_weights, 
        use_jaw_rotation=use_jaw_rotation
    ).to(device)
    _, pred_verts_3d = fitter.fit(
        verts_3d,
        n_steps=n_steps,
        w_shape_reg=w_shape_reg,
        w_expr_reg=w_expr_reg,
    )

    # fix eye rotations
    for frame_id in range(verts_3d.shape[0]):
        if gaze_directions[frame_id] is None:
            eye_rot = np.zeros(3)
        else:
            yaw, pitch = gaze_directions[frame_id][0]
            eye_rot = compute_eyeball_rotation(
                yaw.cpu().numpy(),
                pitch.cpu().numpy(),
                cam_rt[:3, :3],
                cam_rt[:3, 3],
                fitter.rot.data[frame_id].detach().cpu().numpy(),
                fitter.tra.data[frame_id].detach().cpu().numpy(),
            )
        fitter.eye_rot.data[frame_id] = torch.from_numpy(eye_rot).float().to(device)

    clamp_factor = fitter.eye_rot.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    fitter.eye_rot.data = fitter.eye_rot.data / clamp_factor * clamp_factor.clamp(max=1.)

    if smooth_eye_rotations:
        fitter.eye_rot.data = torch.from_numpy(gaussian_filter1d(
            fitter.eye_rot.data.cpu().numpy(), sigma=2, axis=0
        )).float()

    # rerun flame
    pred_verts_3d = fitter.forward()["verts_3d"]

    return (
        fitter.export_results(), 
        pred_verts_3d.detach().cpu().numpy(), 
        flame.template_faces.cpu().numpy(),
    )


def convert_calibration(
    tracking_resolution,
    crop_box,
    k, 
):
    """
    Convert camera intrinsics from cropped and resized space to original resolution space.

    Parameters:
    orig_resolution (tuple): (H_orig, W_orig)
    tracking_resolution (tuple): (H_track, W_track)
    crop_box (tuple): (x0, y0, crop_w, crop_h) in original resolution
    rt (np.ndarray): [4, 4] extrinsic matrix (unchanged here)
    k (np.ndarray): [3, 3] intrinsics in tracking resolution

    Returns:
    new_k (np.ndarray): [3, 3] intrinsics in original resolution
    rt (np.ndarray): unchanged extrinsics
    """
    x0, y0, x1, y1 = crop_box
    crop_w = x1 - x0
    crop_h = y1 - y0
    H_track, W_track = tracking_resolution

    # 1. Compute the scale factor used for resizing the cropped image to tracking resolution
    scale_x = crop_w / W_track
    scale_y = crop_h / H_track

    # 2. Undo the scale (tracking → crop)
    k[0, :] *= scale_x  # fx, cx
    k[1, :] *= scale_y  # fy, cy

    # 3. Undo the crop (crop → original)
    k[0, 2] += x0  # cx
    k[1, 2] += y0  # cy

    return k


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


def main(args):
    tracking_path = Path(args.tracking_path)
    preprocess_path = Path(args.preprocess_path)

    pixel_frames = sorted(list((tracking_path / "checkpoint").glob("*.*")))
    assert len(pixel_frames) > 0, "No Pixel3DMM results found. Maybe it failed?"

    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)

    video_path = Path(args.video_path)

    assert video_path.exists(), "Input video does not exist"
    if video_path.is_dir():
        frame_reader = FrameReader(video_path)
        shutil.copytree(video_path, output_path / "images" / "cam0", dirs_exist_ok=True)
        is_video = False
    else:
        frame_reader = VideoReader(str(video_path))
        shutil.copy(video_path, output_path / "images" / f"cam0{video_path.suffix}")
        is_video = True

    print("Eye tracking with L2CS and background segmentation with RobustVideoMatting")
    l2cs_tracker = L2CSTracker(device=args.device)

    matting_model = MattingNetwork()
    matting_model.load_state_dict(torch.load("data/weights/rvm/rvm_mobilenetv3.pth"))
    matting_model.eval()
    matting_model.to(args.device)

    output_bg_path = output_path / "bg" / "cam0"
    output_bg_path.mkdir(exist_ok=True, parents=True)

    gaze_directions = []
    rec = [None] * 4

    for frame_id, frame_img in enumerate(tqdm(frame_reader)):
        if not isinstance(frame_img, np.ndarray):
            frame_img = frame_img.asnumpy()

        frame_img = torch.from_numpy(frame_img)[None] / 255.

        if args.enable_gaze_tracking:
            with torch.no_grad():
                gaze = l2cs_tracker.process(frame_img)
        else:
            gaze = None
        gaze_directions.append(gaze)

        downsample_ratio = auto_downsample_ratio(*frame_img.shape[2:])

        src = frame_img.permute(0, 3, 1, 2).to(args.device)
        with torch.no_grad():
            fgr, pha, *rec = matting_model(src, *rec, downsample_ratio)
        if not is_video:
            # reset recurrent stuff from matting model
            rec = [None] * 4

        cv2.imwrite(
            str(output_bg_path / f"{frame_id:04d}.png"), 
            (pha[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
        )

    print("Converting Pixel3DMM results to FlowFace format.")
    print(f"Input video: {video_path}")
    print(f"Tracking path: {tracking_path}")
    print(f"Output path: {output_path}")
    print(f"Number of frames: {len(pixel_frames)}")

    crop_box = np.load(preprocess_path / "crop_ymin_ymax_xmin_xmax.npy")
    crop_box = crop_box[[2, 0, 3, 1]]

    all_vertices = []

    print("Loading input video and tracking results")
    for frame_id, frame_path in enumerate(tqdm(pixel_frames)):
        frame_name = frame_path.stem

        mesh_path = tracking_path / "mesh" / f"{frame_name}.ply"

        mesh = trimesh.load(mesh_path)
        vertices = mesh.vertices

        frame_info = torch.load(frame_path, weights_only=False)

        # transform vertices with base head transformation
        vertices = frame_info["flame"]["R_rotation_matrix"][0] @ vertices.T + frame_info["flame"]["t"].T
        vertices = vertices.T

        if frame_id == 0:
            rt = np.eye(4)
            rt[:3, :3] = frame_info["camera"]["R_base_0"][0]
            rt[:3, 3] = frame_info["camera"]["t_base_0"][0]
            rt = OPENCV2PYTORCH3D.inverse().numpy() @ rt
            k = np.eye(3)
            size = 256
            k[:2, 2] = (frame_info["camera"]["pp"][0] + 1.) * (size / 2 + 0.5)
            k[0, 0] = frame_info["camera"]["fl"][0, 0] * size
            k[1, 1] = frame_info["camera"]["fl"][0, 0] * size

            tracking_resolution = frame_info["img_size"]
            orig_resolution = frame_reader[frame_id].shape[:2]

            k_converted = convert_calibration(
                tracking_resolution,
                crop_box,
                k,
            )

        all_vertices.append(vertices)

    out_flame = {}

    # write camera calibration
    out_flame["fx"] = k_converted[0, 0][None, None].astype(np.float32)
    out_flame["fy"] = k_converted[1, 1][None, None].astype(np.float32)
    out_flame["cx"] = k_converted[0, 2][None, None].astype(np.float32)
    out_flame["cy"] = k_converted[1, 2][None, None].astype(np.float32)
    out_flame["extr"] = rt[None].astype(np.float32)
    out_flame["resolutions"] = np.array([orig_resolution])
    out_flame["camera_order"] = ["cam0"]

    verts_3d = np.stack(all_vertices, axis=0)

    fit, pred_verts_3d, template_faces = fit_flame(
        verts_3d,
        gaze_directions,
        OPENCV2PYTORCH3D.numpy() @ rt,
        use_jaw_rotation=False,
        n_shape_params=150, 
        n_expr_params=65, 
        device=args.device,
        n_steps=8000,
        w_shape_reg=1e-4, # 6
        w_expr_reg=1e-4, # 6
        smooth_eye_rotations=is_video,
    )

    converted_mesh_dir = tracking_path / "flowface_mesh"
    converted_mesh_dir.mkdir(exist_ok=True)

    for frame_id, frame_path in enumerate(pixel_frames):
        frame_name = frame_path.stem

        trimesh.Trimesh(
            pred_verts_3d[frame_id], faces=template_faces
        ).export(converted_mesh_dir / f"{frame_name}.ply")
    
    out_flame["rot"] = fit["rot"]
    out_flame["tra"] = fit["tra"]
    out_flame["shape"] = fit["shape"]
    out_flame["expr"] = fit["expr"]
    out_flame["neck_rot"] = fit["neck_rot"]
    out_flame["eye_rot"] = fit["eye_rot"]  # TODO smooth eye rotations

    np.savez(output_path / "fit.npz", **out_flame)

    n_frames = len(pixel_frames)

    np.random.seed(123)
    selected_ids = np.random.permutation(np.arange(n_frames))[:args.max_n_ref]
    selected_ids = sorted(selected_ids)
    reference_info = [["cam0", selected_id.item()] for selected_id in selected_ids]

    with open(output_path / "reference_images.json", 'w') as f:
        json.dump(reference_info, f, indent=4)

    if is_video:
        # If it is a video, that means we can use it for animation. 
        # Save the camera trajectory and create a corresponding camera orbit.
        n_orbit_frames = n_frames
        fps = frame_reader.get_avg_fps()

        print("saving camera trajectories")
        trajectory = {
            "extr": out_flame["extr"].repeat(n_orbit_frames, axis=0),
            "fx": out_flame["fx"].repeat(n_orbit_frames, axis=0),
            "fy": out_flame["fy"].repeat(n_orbit_frames, axis=0),
            "cx": out_flame["cx"].repeat(n_orbit_frames, axis=0),
            "cy": out_flame["cy"].repeat(n_orbit_frames, axis=0),
            "resolution": orig_resolution,
            "fps": fps,
        }
        np.savez(output_path / "cam_static.npz", **trajectory)

        # Create orbit trajectory
        t = np.arange(n_orbit_frames) / fps / ORBIT_PERIOD
        yaw_angles = np.cos(t * 2 * np.pi) * ORBIT_AMPLITUDE_YAW
        pitch_angles = np.sin(t * 2 * np.pi) * ORBIT_AMPLITUDE_PITCH
        for i in range(n_orbit_frames):
            target = out_flame["tra"][0].copy()
            target[1:] = -target[1:]
            trajectory["extr"][i] = pivot_camera_intrinsic(
                trajectory["extr"][i],
                target,
                [yaw_angles[i], pitch_angles[i]]
            )
        np.savez(output_path / "cam_orbit.npz", **trajectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="path to the input video (can be folder of frames or video)",
    )
    parser.add_argument(
        "--tracking_path",
        type=str,
        required=True,
        help="path to tracking output of Pixel3DMM tracker",
    )
    parser.add_argument(
        "--preprocess_path",
        type=str,
        required=True,
        help="path to preprocess output of Pixel3DMM",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="path where the converted (FlowFace format) data will be saved",
    )
    parser.add_argument(
        "--max_n_ref",
        type=int,
        default=100,
        help="maximum number of reference frames",
    )
    parser.add_argument(
        "--enable_gaze_tracking",
        type=int,
        default=1,
        help="whether to enable gaze tracking (if False, eyeball rotation will be zero)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device",
    )
    args = parser.parse_args()
    main(args)

# python scripts/pixel3dmm/convert_to_flowface.py --video_path examples/input/felix/images/cam0/ --tracking_path examples/pixel3dmm_tracking/cam0_nV1_noPho_uv2000.0_n1000.0/ --preprocess_path examples/pixel3dmm_tracking/cam0/ --output_path examples/input/felix_converted/ 

# python cap4d/inference/generate_images.py --config_path configs/generation/debug.yaml --reference_data_path examples/input/felix_converted/ --output_path examples/debug_output/felix_converted/

# Create hugging face demo
