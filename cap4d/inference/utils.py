from collections import defaultdict
import os

import einops
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2

from controlnet.ldm.util import instantiate_from_config
from cap4d.mmdm.mmdm import MMLDM

    
def to_device(batch, device="cuda"):
    for key in batch:
        if isinstance(batch[key], dict):
            batch[key] = to_device(batch[key])
        elif not isinstance(batch[key], list):
            batch[key] = batch[key].to(device)

    return batch


def log_cond(module, batch):
    cond_model = module.cond_stage_model
    cond_key = module.control_key

    c_cond = cond_model(batch[cond_key], unconditional=False)
    enc_vis = cond_model.get_vis(c_cond["pos_enc"])

    for key in enc_vis:
        vis = enc_vis[key]
        b_ = vis.shape[0]
        vis = einops.rearrange(vis, 'b t h w c -> (b t) c h w')
        vis = F.interpolate(vis, scale_factor=8., mode="nearest")
        vis = vis.clamp(-1., 1.)
        enc_vis[key] = einops.rearrange(vis, '(b t) c h w -> (b t) h w c', b=b_)

    return enc_vis


def load_model(ckpt_path):
    list_of_files = list((ckpt_path / "checkpoints").glob("*.ckpt"))
    latest_file = max(list_of_files, key=os.path.getctime)
    print("Loading model using checkpoint", latest_file)
    weight_path = latest_file

    # load modified model
    config_path = ckpt_path / "config_dump.yaml"
    config = OmegaConf.load(config_path)
    print(f'Loaded model config from [{config_path}]')
    model: MMLDM = instantiate_from_config(config.model).cpu()

    print("Loading state dict")
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_condition_from_dataloader(model, dataloader, device):
    cond_frames = defaultdict(list)
    uncond_frames = defaultdict(list)
    cond_vis_frames = defaultdict(list)
    flame_params = []

    for frame_id, batch in enumerate(tqdm(dataloader)):
        batch = to_device(batch, device=device)

        # get conditioning from data batch
        z, c = model.get_input(batch, model.first_stage_key, force_conditional=True)

        # store conditioning for later
        for key in c["c_concat"][0]:
            c_cond = einops.rearrange(c["c_concat"][0][key], 'b t ... -> (b t) ...')
            c_uncond = einops.rearrange(c["c_uncond"][0][key], 'b t ... -> (b t) ...')
            cond_frames[key].append(c_cond.cpu())
            uncond_frames[key].append(c_uncond.cpu())
        
        # store conditioning visualization for later
        cond_vis = log_cond(model, batch)
        for key in cond_vis:
            cond_vis_frames[key].append(cond_vis[key].cpu())
        
        # store the flame and camera parameters for later
        for b in range(batch["flame_params"]["fx"].shape[0]):
            flame_dict = {}
            for key in batch["flame_params"]:
                flame_dict[key] = batch["flame_params"][key][b].cpu().numpy()
            flame_params.append(flame_dict)

    return {
        "cond_frames": cond_frames,
        "uncond_frames": uncond_frames,
        "cond_vis_frames": cond_vis_frames,
        "flame_params": flame_params,
    }


def save_visualization(vis_frames, output_dir):
    condition_base_dir = output_dir / "condition_vis"
    condition_base_dir.mkdir(exist_ok=True)

    for key in vis_frames:
        for frame_id, vis_img in enumerate(vis_frames[key]):
            out_dir = condition_base_dir / f"{key}"
            out_dir.mkdir(exist_ok=True)

            vis_img = vis_img[0]
            cv2.imwrite(
                str(out_dir / f"{frame_id:05d}.jpg"),
                (((vis_img[..., [2, 1, 0]].cpu().numpy() + 1.) / 2.) * 255).astype(np.uint8),
            )


def save_flame_params(flame_params, output_dir):
    out_flame_dir = output_dir / "flame"
    out_flame_dir.mkdir(exist_ok=True)
    
    for frame_id, flame_item in enumerate(flame_params):
        np.savez(out_flame_dir / f"{frame_id:05d}.npz", **flame_item)


def convert_and_save_latent_images(latents, model, device, output_dir):
    out_img_dir = output_dir / "images"
    out_img_dir.mkdir(exist_ok=True)

    for i in range(latents.shape[0]):
        # Convert latent to RGB
        x_samples = model.decode_first_stage(latents[None, [i]].to(device))[0, 0]
        img = ((x_samples + 1.) / 2.).clip(0., 1.)
        img = img.permute(1, 2, 0).cpu().numpy() * 255.
        out_img_path = out_img_dir / f"{i:05d}.png"
        success = cv2.imwrite(str(out_img_path), img[..., [2, 1, 0]].astype(np.uint8))
        assert success, f"failed to save image to {out_img_path}"


def find_number_of_generated_images(n_gen_default, n_ref, R_max, V):
    R = min(n_ref, R_max)
    missing_frames = n_gen_default % (V - R)
    n_gen_new = n_gen_default + missing_frames

    if missing_frames > 0:
        print(f"WARNING: Default number of generated images {n_gen_default} incompatible with R={R}, adjusting the number to {n_gen_new}")

    return n_gen_new
    