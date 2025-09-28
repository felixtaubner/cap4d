
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import face_alignment
from l2cs import Pipeline

from scipy.spatial.transform import Rotation as R


def resize_to_max_res(image: torch.Tensor, max_resolution: int):
    """
    Resizes the image so that its highest resolution is lower than max_resolution.

    Parameters
    ----------
    image: torch.Tensor (B, C, H, W)
        the image to scale
    max_resolution: int
        maximum resolution of the highest resolution of the image

    Returns
    -------
    scaled_image: torch.Tensor (B, C, H, W)
    factor: the factor by which the image was scaled
    """

    factor = 1.0
    max_shape = 2 if image.shape[2] > image.shape[3] else 3
    if image.shape[max_shape] > max_resolution:
        factor = max_resolution / image.shape[max_shape]
        image = F.interpolate(
            image,
            (
                int(image.shape[2] * factor),
                int(image.shape[3] * factor),
            ),
        )
    return image, factor


def compute_eyeball_rotation(yaw, pitch, R_cam, T_cam, r_head, t_head):
    """
    Compute eyeball rotation in FLAME head-local coordinates from gaze direction.

    Parameters:
        yaw, pitch: float (in degrees)
            Gaze direction in camera space (yaw left/right, pitch up/down)
        R_cam: (3, 3) numpy array
            Rotation matrix from world → camera
        T_cam: (3,) or (3, 1) numpy array
            Translation vector from world → camera
        r_head: (3,) numpy array
            Rotation vector from head-local → world
        t_head: (3,) or (3, 1) numpy array
            Translation vector from head-local → world

    Returns:
        eye_rot_mat: (3, 3) numpy array
            Rotation matrix that transforms FLAME eyeball's +Z to gaze direction
    """
    R_head = R.from_rotvec(r_head).as_matrix()

    def gaze_direction_from_yaw_pitch(yaw, pitch):
        # yaw = np.radians(yaw)
        # pitch = np.radians(pitch)
        x = np.sin(yaw) * np.cos(pitch)
        y = -np.sin(pitch)
        z = np.cos(yaw) * np.cos(pitch)
        return np.array([x, y, z])

    def rotation_to_vector(src, dst):
        src = src / np.linalg.norm(src)
        dst = dst / np.linalg.norm(dst)
        v = np.cross(src, dst)
        c = np.dot(src, dst)
        if c < -0.999999:
            axis = np.cross(src, [1, 0, 0])
            if np.linalg.norm(axis) < 1e-5:
                axis = np.cross(src, [0, 1, 0])
            axis = axis / np.linalg.norm(axis)
            return R.from_rotvec(np.pi * axis).as_matrix()
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2 + 1e-8))

    # Step 1: Gaze vector in camera space
    gaze_dir_cam = gaze_direction_from_yaw_pitch(yaw, pitch)

    # Step 2: Convert to world space
    R_cam = R_cam.astype(np.float64)
    T_cam = T_cam.reshape(3, 1)
    R_cam_to_world = R_cam.T
    gaze_dir_world = R_cam_to_world @ gaze_dir_cam

    # Step 3: Convert to head-local space
    R_head = R_head.astype(np.float64)
    t_head = t_head.reshape(3, 1)
    R_world_to_head = R_head.T
    gaze_dir_head = R_world_to_head @ gaze_dir_world

    # Step 4: Compute rotation from +Z to gaze direction
    forward = np.array([0, 0, -1])
    eye_rot_mat = rotation_to_vector(forward, gaze_dir_head)

    return R.from_matrix(eye_rot_mat).as_rotvec()


class L2CSTracker:
    def __init__(
        self,
        device="cuda",
        gaze_weight_path="data/weights/l2cs/L2CSNet_gaze360.pkl"
    ) -> None:
        super().__init__()

        self.device = device

        self.fa = face_alignment.FaceAlignment(
            3, flip_input=False, face_detector="sfd", device=device, 
            face_detector_kwargs={"filter_threshold": 0.9}
        )
        # self.retina = RetinaFace()
        self.gaze_trafo = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.gaze_model = Pipeline(
            weights=gaze_weight_path,
            arch='ResNet50',
            device=torch.device(device),
            include_detector=False,
        )
        self.gaze_model.softmax = torch.nn.Softmax(dim=1)  # this is required because of a bug in their code
        self.gaze_model.idx_tensor = torch.FloatTensor([idx for idx in range(90)]).to(device)


    def predict_gaze(self, l2cs_input):
        gaze_yaw, gaze_pitch = self.gaze_model.model(l2cs_input)

        # softmax = torch.nn.Softmax()
        pitch_pmf = F.softmax(gaze_pitch, dim=1)
        yaw_pmf = F.softmax(gaze_yaw, dim=1)

        # convert angles into rad
        value_tensor = (self.gaze_model.idx_tensor * 4 - 180) * torch.pi / 180.
        
        pitch_mean = torch.sum(pitch_pmf * value_tensor[None], dim=-1)
        yaw_mean = torch.sum(yaw_pmf * value_tensor[None], dim=-1)

        gaze_mean = torch.stack([yaw_mean, pitch_mean], dim=-1)

        return gaze_mean


    def process(self, images: torch.Tensor):
        images = images.permute(0, 3, 1, 2).to(self.device)  # (B, H, W, C -> B, C, H, W)
        small_imgs, scale_factor = resize_to_max_res(images, 512)
        bboxes_stack = self.fa.face_detector.detect_from_batch(small_imgs * 255)

        l2cs_images = []
        bbox_valids = []
        for i in range(images.shape[0]):
            img = images[i]
            bboxes = bboxes_stack[i]

            bbox_valid = True
            if len(bboxes) == 0:
                bbox_valid = False
                print("WARNING, no face detected!")
                bbox = np.array([0, 0, img.shape[2], img.shape[1]])
            else:
                bboxes = sorted(bboxes, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]))
                if len(bboxes) > 1:
                    print("WARNING, multiple faces detected!")
                bbox = bboxes[-1] / scale_factor  # choose largest bbox

            l2cs_img = img[:, max(0, int(bbox[1])):int(bbox[3]), max(0, int(bbox[0])):int(bbox[2])]
            l2cs_img = self.gaze_trafo(l2cs_img[None])[0]

            l2cs_images.append(l2cs_img)
            bbox_valids.append(bbox_valid)

        l2cs_images = torch.stack(l2cs_images, dim=0)
    
        valid_mask = torch.tensor(bbox_valids).to(self.device)

        gaze_mean = self.predict_gaze(l2cs_images)

        # set gaze zero when no face detected
        gaze_mean[~valid_mask] = 0.

        return gaze_mean
