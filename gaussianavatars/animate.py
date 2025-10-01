#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from pathlib import Path
from os import makedirs
import subprocess
import concurrent.futures

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

from argparse import ArgumentParser
from gaussianavatars.utils.system_utils import searchForMaxIteration
from gaussianavatars.utils.general_utils import safe_state
from gaussianavatars.gaussian_renderer.gsplat_renderer import render
from gaussianavatars.scene.scene import Scene
from gaussianavatars.scene.cap4d_gaussian_model import CAP4DGaussianModel
from gaussianavatars.utils.export_utils import PlyWriter


def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            if data.dtype == torch.long:
                data = data[0].to("cpu", torch.uint16).numpy()
            else:
                data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")


def frames_to_video(
    frame_dir,
    output_path,
    fps,
):
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-f", "image2",
        "-pattern_type", "glob",
        "-i", f"{frame_dir}/*.png",
        "-crf", "18",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        f"{output_path}"
    ]

    # Run the command with real-time output
    subprocess.run(cmd, check=True)


def render_sequence(args):
    model_path = Path(args.model_path)
    target_paths = {
        "animation_path": args.target_animation_path,
        "cam_trajectory_path": args.target_cam_trajectory_path,
    }

    avatar_config = OmegaConf.load(model_path / "config_dump.yaml")

    gaussians = CAP4DGaussianModel(avatar_config["model_params"])
    gaussians.eval()
    
    scene = Scene(
        source_paths=None,
        target_paths=target_paths,
        model_path=model_path,
        gaussians=gaussians, 
        shuffle=False,
    )

    loaded_iter, chkpt_path = searchForMaxIteration(model_path)

    assert loaded_iter is not None, f"No valid checkpoint found in {model_path}"
    print("Loading trained model at iteration {}".format(loaded_iter))
    (model_weights, first_iter) = torch.load(chkpt_path, weights_only=False)
    gaussians.restore(model_weights)

    bg_color = [1, 1, 1]  # force white background
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    output_path = Path(args.output_path)
    makedirs(output_path, exist_ok=True)
    render_path = output_path / "renders"
    if args.render_alpha:
        render_alpha_path = output_path / "renders_alpha"
        makedirs(render_alpha_path, exist_ok=True)
    if args.render_depth:
        render_depth_path = output_path / "renders_depth"
        makedirs(render_depth_path, exist_ok=True)
    if args.export_ply:
        ply_writer = PlyWriter(compress=args.compress_ply)

    makedirs(render_path, exist_ok=True)

    views_loader = DataLoader(
        scene.getTgtCameras(), 
        batch_size=None, 
        shuffle=False, 
        num_workers=8,
    )
    max_threads = 4
    worker_args = []

    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):

        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(view.timestep)

        render_out = render(
            view, 
            gaussians, 
            background, 
            compute_depth=args.render_depth
        )
        
        rendering = render_out["render"]

        if args.export_ply:
            ply_writer.update(gaussians)

        path2data = {}
        path2data[Path(render_path) / f'{idx:05d}.png'] = rendering
        if args.render_alpha:
            alpha = render_out["alpha"]
            blended_rendering = torch.cat([rendering, alpha], dim=0)
            path2data[Path(render_alpha_path) / f'{idx:05d}.png'] = blended_rendering
        if args.render_depth:
            depth = render_out["depth"]
            depth = (depth * 1000.).long()
            path2data[Path(render_depth_path) / f'{idx:05d}.png'] = depth

        worker_args.append([path2data])

        if len(worker_args) == max_threads or idx == len(views_loader)-1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, *args) for args in worker_args]
                concurrent.futures.wait(futures)
            worker_args = []

    if args.export_ply:
        print("Exporting animation...")
        ply_writer.save_ply(output_path / "exported_animation.ply")

    # load fps
    fps = 24  # default fps
    if args.target_cam_trajectory_path is not None:
        camera_trajectory = dict(np.load(args.target_cam_trajectory_path))
        if "fps" in camera_trajectory:
            fps = camera_trajectory["fps"]

    frames_to_video(render_path, output_path / "renders.mp4", fps=fps)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Render the reconstructed avatar with a " \
    "driving animation and camera trajectory.")
    parser.add_argument('--model_path', type=str, help="Path to directory " \
                                            "where the gaussian avatar model is saved.")
    parser.add_argument('--target_animation_path', type=str, help="Path to driving animation (fit.npz).")
    parser.add_argument('--target_cam_trajectory_path', type=str, default=None,
                        help="Path to driving camera trajectory (*.npz). " \
                        "This trajectory describes per frame camera intrinsics and " \
                        "extrinsics - the extrinsics are expressed relative to the " \
                        "extrinsics of the input animation. If this is not given," \
                        "the camera of the driving sequence will be used.")
    parser.add_argument('--output_path', type=str, required=True,
                        help="Path to directory where the animation outputs " \
                        "(frames, video and optionally ply) will be saved.")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_alpha", type=int, default=False)
    parser.add_argument("--render_depth", type=int, default=False)
    parser.add_argument("--export_ply", type=int, default=True, 
                        help="save baked ply animation for web rendering")
    parser.add_argument("--compress_ply", type=int, default=False, 
                        help="compress baked ply animation at the cost of quality (jittering)")

    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    with torch.no_grad():
        render_sequence(args)
