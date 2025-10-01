from pathlib import Path
import argparse
import copy
import shutil

import torch
from torch.utils.data import  DataLoader
import gc
from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from cap4d.mmdm.sampler import StochasticIOSampler
from cap4d.inference.data.reference_data import ReferenceDataset
from cap4d.inference.data.generation_data import GenerationDataset
from cap4d.inference.utils import (
    get_condition_from_dataloader,
    load_model,
    save_visualization,
    save_flame_params,
    convert_and_save_latent_images,
    find_number_of_generated_images,
)


@torch.no_grad()
def main(args):
    ref_data_path = Path(args.reference_data_path)

    gen_config_path = Path(args.config_path)
    gen_config = OmegaConf.load(gen_config_path)

    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    output_ref_path = output_path / "reference_images"
    output_ref_path.mkdir(exist_ok=True)
    output_gen_path = output_path / "generated_images"
    output_gen_path.mkdir(exist_ok=True)

    seed_everything(gen_config["seed"])

    shutil.copy(gen_config_path, output_path / "mmdm_config_dump.yaml")

    # create reference dataloaders
    print("Creating dataloaders")
    refset = ReferenceDataset(ref_data_path, gen_config["resolution"])
    ref_dataloader = DataLoader(refset, num_workers=1, batch_size=1, shuffle=False)

    n_gen_samples = find_number_of_generated_images(
        gen_config["generation_data"]["n_samples"], 
        len(ref_dataloader),
        gen_config["R_max"],
        gen_config["V"],
    )

    # create generation dataloaders
    genset = GenerationDataset(
        generation_data_path=gen_config["generation_data"]["data_path"],
        reference_flame_item=refset.flame_list[0],
        n_samples=n_gen_samples,
        resolution=gen_config["resolution"],
        yaw_range=gen_config["generation_data"]["yaw_range"],
        pitch_range=gen_config["generation_data"]["pitch_range"],
        expr_factor=gen_config["generation_data"]["expr_factor"],
    )
    gen_dataloader = DataLoader(genset, num_workers=1, batch_size=1, shuffle=False)

    # load models
    model = load_model(Path(gen_config["ckpt_path"]))

    device_model_map = {}
    first_rank_model = None
    first_rank_device = None
    if "cuda" in args.device:
        for cuda_id in range(torch.cuda.device_count()):
            dev_key = f"cuda:{cuda_id}"
            device_model_map[dev_key] = copy.deepcopy(model).to(dev_key)
            if first_rank_model is None:
                first_rank_model = device_model_map[dev_key]
                first_rank_device = dev_key
    else:
        device_model_map[args.device] = model.to(args.device)
        first_rank_model = device_model_map[args.device]
        first_rank_device = args.device
        # model.cond_stage_model.device = "cpu"

    print(f"Done loading model.")

    # load all reference frames and create conditioning (and unconditional) images for each
    print(f"Loading reference dataset from {ref_data_path}")
    ref_data = get_condition_from_dataloader(
        first_rank_model,
        ref_dataloader,
        args.device,
    )

    # load all generation frames and create conditioning images for each
    print(f"Loading generation dataset from {gen_config['generation_data']['data_path']}")
    gen_data = get_condition_from_dataloader(
        first_rank_model,
        gen_dataloader,
        args.device,
    )

    if args.visualize_conditioning:
        print("Saving visualization of conditioning images")
        save_visualization(ref_data["cond_vis_frames"], output_ref_path)
        save_visualization(gen_data["cond_vis_frames"], output_gen_path)

    print("Saving flame parameters")
    save_flame_params(ref_data["flame_params"], output_ref_path)
    save_flame_params(gen_data["flame_params"], output_gen_path)

    gc.collect()

    for key in ref_data["cond_frames"]:
        ref_data["cond_frames"][key] = torch.cat(ref_data["cond_frames"][key], dim=0)
        ref_data["uncond_frames"][key] = torch.cat(ref_data["uncond_frames"][key], dim=0)
        gen_data["cond_frames"][key] = torch.cat(gen_data["cond_frames"][key], dim=0)
        gen_data["uncond_frames"][key] = torch.cat(gen_data["uncond_frames"][key], dim=0)

    # Sample with Stochastic I/O Conditioning:
    print(f"Generating images on {len(list(device_model_map))} devices with stochastic I/O.")
    stochastic_io_sampler = StochasticIOSampler(device_model_map)

    z_gen = stochastic_io_sampler.sample(
        S=gen_config["n_ddim_steps"],
        ref_cond=ref_data["cond_frames"],
        ref_uncond=ref_data["uncond_frames"],
        gen_cond=gen_data["cond_frames"],
        gen_uncond=gen_data["uncond_frames"],
        latent_shape=(4, gen_config["resolution"] // 8, gen_config["resolution"] // 8),
        V=gen_config["V"],
        R_max=gen_config["R_max"],
        cfg_scale=gen_config["cfg_scale"],
    )

    print("Done generating.")

    gc.collect()
    z_gen = z_gen.cpu()
    z_ref = ref_data["cond_frames"]["z_input"].cpu()

    print(f"Saving reference images to {output_ref_path}/images")
    convert_and_save_latent_images(z_ref, first_rank_model, first_rank_device, output_ref_path)
    print(f"Saving generated images to {output_gen_path}/images")
    convert_and_save_latent_images(z_gen, first_rank_model, first_rank_device, output_gen_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="path to generation config file",
    )
    parser.add_argument(
        "--reference_data_path",
        type=str,
        required=True,
        help="path to reference json file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="path to output",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="inference device",
    )
    parser.add_argument(
        "--visualize_conditioning",
        type=int,
        default=1,
        help="whether to save visualizations of conditioning images",
    )
    args = parser.parse_args()
    main(args)
