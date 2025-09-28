import contextlib
from pathlib import Path
from typing import Dict

import numpy as np
import einops
import cv2
from decord import VideoReader
from scipy.spatial.transform import Rotation as R

from cap4d.flame.flame import CAP4DFlameSkinner, compute_flame


CROP_MARGIN = 0.2


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def crop_image(
    img: np.ndarray, 
    crop_box: np.ndarray, 
    bg_value=0,
) -> np.ndarray:
    """
    Crop image with the provided crop ranges. If crop box out of image range,
    corresponding pixels will be padded black
    """
    img_h = img.shape[0]
    img_w = img.shape[1]
    crop_h = crop_box[3] - crop_box[1]
    crop_w = crop_box[2] - crop_box[0]
    x_start = max(0, -crop_box[0])
    x_end = max(0, crop_box[2] - img_w)
    y_start = max(0, -crop_box[1])
    y_end = max(0, crop_box[3] - img_h)
    cropped_img = np.ones((crop_h, crop_w, *img.shape[2:]), dtype=img.dtype) * bg_value
    cropped_img[y_start : crop_h - y_end, x_start : crop_w - x_end, ...] = img[
        crop_box[1] + y_start : crop_box[3] - y_end,
        crop_box[0] + x_start : crop_box[2] - x_end,
        ...,
    ]

    return cropped_img


def rescale_image(
    img: np.ndarray,
    target_resolution: int,
):
    interpolation_mode = cv2.INTER_LINEAR
    if target_resolution < img.shape[0]:
        interpolation_mode = cv2.INTER_AREA
    
    img = cv2.resize(img, (target_resolution, target_resolution), interpolation=interpolation_mode)

    return img


def apply_bg(
    img: np.ndarray,
    bg_weights: np.ndarray,
    bg_color: np.ndarray = np.array([255, 255, 255]),
):
    bg_weights = bg_weights / 255.

    bg_img = bg_color[None, None]
    img = bg_img * (1. - bg_weights) + img * bg_weights

    return img


def verts_to_pytorch3d(
    verts_2d: np.ndarray,
    crop_box: np.ndarray,
):
    """
    convert vertex 2D coordinates to pytorch3d screen space convention
    """
    verts_2d[..., 0] = -((verts_2d[..., 0] - crop_box[..., 0]) / (crop_box[..., 2] - crop_box[..., 0]) * 2. - 1.)
    verts_2d[..., 1] = -((verts_2d[..., 1] - crop_box[..., 1]) / (crop_box[..., 3] - crop_box[..., 1]) * 2. - 1.)

    return verts_2d


def get_square_bbox(
    bbox: np.ndarray,
    border_margin: float = 0.1,
    mode: str = "max",  # min or max
):
    """
    Square crops the image with a specified bounding box.
    The image size will be squared, adjusted to the bounding box and min_border
    width.

    Parameters
    ----------
    img_shape: tuple[int, int]
        the shape of the image (h, w)
    bbox: np.ndarray
        the face bounding box

    Returns
    -------
    crop_box: tuple[int, int, int, int]
        the index ranges taken from the original image
        x_min, y_min, x_max, y_max
    """

    bbox = bbox.astype(int)

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]
    b_center = ((bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2)
    if mode == "max":
        dim = int(max(bbox_h, bbox_w) // 2.0 * (1.0 + border_margin))
    elif mode == "min":
        dim = int(min(bbox_h, bbox_w) // 2.0 * (1.0 + border_margin))

    return (
        b_center[0] - dim,
        b_center[1] - dim,
        b_center[0] + dim,
        b_center[1] + dim,
    )


def get_bbox_from_verts(verts_2d, vert_mask):
    head_verts = verts_2d[vert_mask]
    head_bbox = [head_verts[..., 0].min(), head_verts[..., 1].min(), head_verts[..., 0].max(), head_verts[..., 1].max()]
    crop_box = get_square_bbox(np.array(head_bbox), border_margin=CROP_MARGIN)

    return np.array(crop_box)


def load_flame_verts_and_cam(
    flame_skinner: CAP4DFlameSkinner,
    flame_item: Dict[str, np.ndarray],
):
    flame_out = compute_flame(flame_skinner, flame_item)

    verts_2d = flame_out["verts_2d"][0, 0]
    offsets_3d = flame_out["offsets_3d"][0]

    intrinsics = np.eye(3)
    intrinsics[0, 0] = flame_item["fx"][0, 0]
    intrinsics[1, 1] = flame_item["fy"][0, 0]
    intrinsics[0, 2] = flame_item["cx"][0, 0]
    intrinsics[1, 2] = flame_item["cy"][0, 0]
    extrinsics = flame_item["extr"][0]

    return verts_2d, offsets_3d, intrinsics, extrinsics


def load_camera_rays(
    crop_box,
    intr,
    extr,
    target_resolution,
):
    downscale_resolution = target_resolution

    scale = downscale_resolution / (crop_box[2] - crop_box[0])
    new_fx = intr[0, 0] * scale
    new_fy = intr[1, 1] * scale
    new_cx = (intr[0, 2] - crop_box[0]) * scale
    new_cy = (intr[1, 2] - crop_box[1]) * scale

    u, v = np.meshgrid(np.arange(downscale_resolution), np.arange(downscale_resolution)) # [H, w]
    
    d = np.stack(((u - new_cx) / new_fx, (v - new_cy) / new_fy, np.ones_like(u)), axis=0)
    d = d / (np.linalg.norm(d, axis=0, keepdims=True) + 1e-8)
    h, w = d.shape[1:]

    # project camera coordinates back to world
    d = einops.rearrange(d, 'v h w -> v (h w)')
    d = np.linalg.inv(extr[:3, :3]) @ d
    d = einops.rearrange(d, 'v (h w) -> v h w', h=h)

    return d  # ray directions


def adjust_intrinsics_crop(fx, fy, cx, cy, bbox, target_resolution):
    scale = target_resolution / (bbox[2] - bbox[0])
    new_fx = fx * scale
    new_fy = fy * scale
    new_cx = (cx - bbox[0]) * scale
    new_cy = (cy - bbox[1]) * scale

    return new_fx, new_fy, new_cx, new_cy


def get_crop_mask(orig_resolution, target_resolution, crop_box):
    crop_mask = np.ones((orig_resolution))
    crop_mask = crop_image(crop_mask, crop_box, bg_value=0)
    crop_mask = rescale_image(crop_mask, target_resolution)

    return crop_mask


class FrameReader:
    def __init__(self, video_path):
        self.frame_list = sorted(list(Path(video_path).glob("*.*")))

    def __len__(self):
        return len(self.frame_list)
    
    def __getitem__(self, index):
        img = cv2.imread(self.frame_list[index])[..., [2, 1, 0]]
        return img


def load_frame(
    video_path: Path,  # path to .mp4 or dir containing frames
    frame_id: np.ndarray,
):
    if (video_path).is_dir():
        video_reader = FrameReader(video_path)
    else:
        video_reader = VideoReader(str(video_path))

    if frame_id >= len(video_reader):
        print(f"WARNING: Frame {frame_id} out of bounds for video with length {len(video_reader)}")
        frame_id = len(video_reader) - 1

    frame_img = video_reader[frame_id]
    if not isinstance(frame_img, np.ndarray):
        frame_img = frame_img.asnumpy()  # if the video reader is a decord reader

    return frame_img


def pivot_camera_intrinsic(extrinsics, target, angles, distance_factor=1.):
    """
    Rotates a camera around a target point.

    Parameters:
    - extrinsics: (4x4) numpy array, world_to_camera transformation matrix.
    - target: (3,) numpy array, target coordinates to pivot around.
    - angles: (3,) array-like, rotation angles (degrees) around X, Y, Z axes.

    Returns:
    - new_extrinsics: (4x4) numpy array, updated world_to_camera transformation.
    """
    extrinsics = np.linalg.inv(extrinsics)

    # Extract rotation and translation from extrinsics
    R_c2w = extrinsics[:3, :3]  # 3x3 rotation matrix
    t_c2w = extrinsics[:3, 3]   # 3x1 translation vector

    # Compute offset vector from target to camera
    v = (t_c2w - target) * distance_factor

    # Compute rotation matrix for given angles
    R_delta = R.from_euler('YX', angles, degrees=True).as_matrix()  # 'yx'

    # Apply intrinsic rotation to the camera's rotation (local frame)
    new_R_c2w = R_c2w @ R_delta

    # Rotate position offset in camera frame as well
    new_v = R_c2w @ R_delta @ np.linalg.inv(R_c2w) @ v
    new_t_c2w = target + new_v

    # Construct new extrinsics
    new_extrinsics = np.eye(4)
    new_extrinsics[:3, :3] = new_R_c2w
    new_extrinsics[:3, 3] = new_t_c2w

    # import pdb; pdb.set_trace()

    return np.linalg.inv(new_extrinsics)
