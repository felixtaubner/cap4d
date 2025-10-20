import numpy as np

from cap4d.datasets.utils import (
    pivot_camera_intrinsic,
    get_head_direction,
    compute_yaw_pitch_to_face_direction,
)

from cap4d.inference.data.inference_data import CAP4DInferenceDataset


def elipsis_sample(yaw_limit, pitch_limit):
    if yaw_limit == 0. or pitch_limit == 0.:
        return 0., 0.
    
    dist = 1.
    while dist >= 1.:
        yaw = np.random.uniform(-yaw_limit, yaw_limit)
        pitch = np.random.uniform(-pitch_limit, pitch_limit)

        dist = np.sqrt((yaw / yaw_limit) ** 2 + (pitch / pitch_limit) ** 2)

    return yaw, pitch


class GenerationDataset(CAP4DInferenceDataset):
    def __init__(
        self, 
        generation_data_path,
        reference_flame_item,
        n_samples=840,
        yaw_range=55,
        pitch_range=20,
        expr_factor=1.0,
        resolution=512,
        downsample_ratio=8,
    ):
        super().__init__(resolution, downsample_ratio)

        self.n_samples = n_samples
        self.yaw_range = yaw_range
        self.pitch_range = pitch_range

        self.flame_dicts = self.init_flame_params(
            generation_data_path,
            reference_flame_item,
            n_samples,
            yaw_range,
            pitch_range,
            expr_factor,
        )

    def init_flame_params(
        self,
        generation_data_path,
        reference_flame_item,
        n_samples,
        yaw_range,
        pitch_range,
        expr_factor,
    ):
        gen_data = dict(np.load(generation_data_path))

        ref_extr = reference_flame_item["extr"]
        ref_shape = reference_flame_item["shape"]
        ref_fx = reference_flame_item["fx"]
        ref_fy = reference_flame_item["fy"]
        ref_cx = reference_flame_item["cx"]
        ref_cy = reference_flame_item["cy"]
        ref_resolution = reference_flame_item["resolutions"]
        ref_rot = reference_flame_item["rot"]
        ref_tra = reference_flame_item["tra"]
        ref_tra_cv = ref_tra.copy()
        ref_tra_cv[:, 1:] = -ref_tra_cv[:, 1:]  # p3d to opencv
        ref_head_dir = get_head_direction(ref_rot)  # in p3d
        ref_head_dir[:, 1:] = -ref_head_dir[:, 1:]  # p3d to opencv

        center_yaw, center_pitch = compute_yaw_pitch_to_face_direction(
            ref_extr[0],
            ref_head_dir[0],
        )

        flame_list = []

        assert n_samples <= len(gen_data["expr"]), "too many samples"
        for expr, eye_rot in zip(gen_data["expr"][:n_samples], gen_data["eye_rot"][:n_samples]):
            yaw, pitch = elipsis_sample(yaw_range, pitch_range)
            yaw += center_yaw
            pitch -= center_pitch  # pitch is flipped for some reason

            rotated_extr = pivot_camera_intrinsic(ref_extr[0], ref_tra_cv[0], [yaw, pitch])

            flame_dict = {
                "shape": ref_shape,
                "expr": expr[None] * expr_factor,
                "eye_rot": eye_rot[None] * expr_factor,
                "rot": ref_rot,
                "tra": ref_tra,
                "extr": rotated_extr[None],
                "resolutions": ref_resolution,
                "fx": ref_fx,
                "fy": ref_fy,
                "cx": ref_cx,
                "cy": ref_cy,
            }
            flame_list.append(flame_dict)

        self.flame_list = flame_list
        self.ref_extr = ref_extr[0]
