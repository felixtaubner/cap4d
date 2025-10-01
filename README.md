# 🧢 CAP4D
Official repository for the paper

**CAP4D: Creating Animatable 4D Portrait Avatars with Morphable Multi-View Diffusion Models**, ***CVPR 2025 (Oral)***.

<a href="https://felixtaubner.github.io/" target="_blank">Felix Taubner</a><sup>1,2</sup>, <a href="https://scholar.google.com/citations?user=KFx-0xIAAAAJ&hl=en" target="_blank">Ruihang Zhang</a><sup>1</sup>, <a href="https://mathieutuli.com/" target="_blank">Mathieu Tuli</a><sup>3</sup>, <a href="https://davidlindell.com/" target="_blank">David B. Lindell</a><sup>1,2</sup>

<sup>1</sup>University of Toronto, <sup>2</sup>Vector Institute, <sup>3</sup>LG Electronics

<a href='https://arxiv.org/abs/2412.12093'><img src='https://img.shields.io/badge/arXiv-2301.02379-red'></a> <a href='https://felixtaubner.github.io/cap4d/'><img src='https://img.shields.io/badge/project page-CAP4D-Green'></a> <a href='#citation'><img src='https://img.shields.io/badge/cite-blue'></a>

![Preview](assets/banner.gif)

TL;DR: CAP4D turns any number of reference images into an animatable avatar. 

## ⚡️ Quick start guide

### 🛠️ 1. Create conda environment and install requirements

```bash
# 1. Clone repo
git clone https://github.com/felixtaubner/cap4d/
cd cap4d

# 2. Create conda environment for CAP4D:
conda create --name cap4d_env python=3.10
conda activate cap4d_env

# 3. Install requirements
pip install -r requirements.txt

# 4. Set python path
export PYTHONPATH=$(realpath "./"):$PYTHONPATH
```
Follow the [instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) and install Pytorch3D. Make sure to install with CUDA support. We recommend to install from source: 

```bash
export FORCE_CUDA=1
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

### 📦 2. Download FLAME and MMDM weights
Setup your FLAME account at the [FLAME website](https://flame.is.tue.mpg.de/index.html) and set the username 
and password environment variables:
```bash
export FLAME_USERNAME=your_flame_user_name
export FLAME_PWD=your_flame_password
```

Download FLAME and MMDM weights using the provided scripts:

```bash 
# 1. Download FLAME blendshapes
# set your flame username and password
bash scripts/download_flame.sh 

# 2. Download CAP4D MMDM weights
bash scripts/download_mmdm_weights.sh
```

If the FLAME download script did not work, download FLAME2023 from the [FLAME website](https://flame.is.tue.mpg.de/index.html) and place `flame2023_no_jaw.pkl` in `data/assets/flame/`.
Then, fix the flame pkl file to be compatible with newer numpy versions:

```bash
python scripts/fixes/fix_flame_pickle.py --pickle_path data/assets/flame/flame2023_no_jaw.pkl
```

### ✅ 3. Check installation with a test run
Run the pipeline in debug settings to test the installation.

```bash
bash scripts/test_pipeline.sh
```

Check if a video is exported to `examples/debug_output/tesla/sequence_00/renders.mp4`.
If it appears to show a blurry cartoon Nicola Tesla, you're all set! 

### 🎬 4. Inference 
Run the provided scripts to generate avatars and animate them with a single script:

```bash
bash scripts/generate_felix.sh
bash scripts/generate_lincoln.sh
bash scripts/generate_tesla.sh
```

The output directories contain exported animations which you can view in real-time.
Open the [real-time viewer](https://felixtaubner.github.io/cap4d/viewer/) in your browser (powered by [Brush](https://github.com/ArthurBrussee/brush/)). Click `Load file` and
upload the exported animation found in `examples/output/{SUBJECT}/animation_{ID}/exported_animation.ply`.

## 🔧 Custom inference

See below for how to run your custom inference on your own reference images/videos and driving videos.

### ⚙️ 1. Run FLAME 3D face tracking

#### 1.1 FlowFace tracking
Coming soon! For now, only generations using the provided identities with precomputed [FlowFace](https://felixtaubner.github.io/flowface/) annotations are supported. 

#### 1.2 Pixel3DMM tracking
Install [Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm) using the provided script. Notice that this is prone to errors due to package version mismatches. Please report any errors as an issue!

```bash
export FLAME_USERNAME=your_flame_user_name
export FLAME_PWD=your_flame_password
export PIXEL3DMM_PATH=$(realpath "../PATH/TO/pixel3dmm")  # set this to where you would like to clone the Pixel3DMM repo (absolute path)
export CAP4D_PATH=$(realpath "./")  # set this to the cap4d directory (absolute path)

bash scripts/install_pixel3Dmm.sh
```

Run tracking and conversion on reference images/videos using the provided script. Note: If input is a directory of frames, it is assumed to be discontinous set of (monocular!) images. If input is a file, it will assume that it is a continous monocular video.

```bash
export PIXEL3DMM_PATH=$(realpath "../PATH/TO/pixel3dmm")
export CAP4D_PATH=$(realpath "./") 

mkdir examples/output/custom/

# For more information on arguments
bash scripts/track_video_pixel3dmm.sh --help

# Process a directory of (reference) images
bash scripts/track_video_pixel3dmm.sh examples/input/felix/images/cam0/ examples/output/custom/reference_tracking/

# Optional: process a driving (or reference) video
bash scripts/track_video_pixel3dmm.sh examples/input/animation/example_video.mp4 examples/output/custom/driving_video_tracking/
```

Notice that results will be slightly worse than with FlowFace tracking, since the MMDM is trained with FlowFace.

### 🖼️ 2. Generate images using MMDM

```bash
# Generate images with single reference image
python cap4d/inference/generate_images.py --config_path configs/generation/default.yaml --reference_data_path examples/output/custom/reference_tracking/ --output_path examples/output/custom/mmdm/
```
Note: the generation script will use all visible CUDA devices. The more available devices, the faster it runs! This will take hours, and requires lots of RAM (ideally > 64 GB) to run smoothly.

### 👤 3. Fit Gaussian avatar 

```bash
python gaussianavatars/train.py --config_path configs/avatar/default.yaml --source_paths examples/output/custom/mmdm/reference_images/ examples/output/custom/mmdm/generated_images/ --model_path examples/output/custom/avatar/ --interval 5000
```

### 🕺 4. Animate your avatar

Once the avatar is generated, it can be animated with the driving video computed in step 1 or the provided animations. 

```bash
# Animate the avatar with provided animation files
python gaussianavatars/animate.py --model_path examples/output/custom/avatar/ --target_animation_path examples/input/animation/sequence_00/fit.npz  --target_cam_trajectory_path examples/input/animation/sequence_00/orbit.npz  --output_path examples/output/custom/animation_00/ --export_ply 1 --compress_ply 0

# Animate the avatar with driving video (computed using Pixel3DMM)
python gaussianavatars/animate.py --model_path examples/output/custom/avatar/ --target_animation_path examples/output/custom/driving_video_tracking/fit.npz  --target_cam_trajectory_path examples/output/custom/driving_video_tracking/cam_static.npz  --output_path examples/output/custom/animation_example/ --export_ply 1 --compress_ply 0
```

The `--target_animation_path` argument contains FLAME expressions and pose, while the (optional) `--target_cam_trajectory_path` argument contains the relative camera trajectory. 

### ⚡️ 5. Full inference

We provide a convenient script to run full inference using your reference images and optionally a driving video.

```bash
export PIXEL3DMM_PATH=$(realpath "../PATH/TO/pixel3dmm")
export CAP4D_PATH=$(realpath "./") 

# Generate avatar with custom input images/videos.
bash scripts/generate_avatar.sh --help
bash scripts/generate_avatar.sh {INPUT_VIDEO_PATH} {OUTPUT_PATH} [{QUALITY}] [{DRIVING_VIDEO_PATH}]

# Example generation with default quality generation with input images and driving video.
bash scripts/generate_avatar.sh examples/input/felix/images/cam0/ examples/output/felix_custom/ default examples/input/animation/example_video.mp4
```

### ✨ 6. View avatar in live viewer

Open the [real-time viewer](https://felixtaubner.github.io/cap4d/viewer/) in your browser (powered by [Brush](https://github.com/ArthurBrussee/brush/)). Click `Load file` and
upload the exported animation found in 
`examples/output/custom/animation_00/exported_animation.ply` or
`examples/output/custom/animation_example/exported_animation.ply`.

## 📚 Related Resources

The MMDM code is based on [ControlNet](https://github.com/lllyasviel/ControlNet). The 4D Gaussian avatar code is based on [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars). Special thanks to the authors for making their code public!

Related work: 
- [CAT3D](https://cat3d.github.io/): Create Anything in 3D with Multi-View Diffusion Models
- [GaussianAvatars](https://shenhanqian.github.io/gaussian-avatars): Photorealistic Head Avatars with Rigged 3D Gaussians
- [FlowFace](https://felixtaubner.github.io/flowface/): 3D Face Tracking from 2D Video through Iterative Dense UV to Image Flow
- [StableDiffusion](https://github.com/Stability-AI/stablediffusion): High-Resolution Image Synthesis with Latent Diffusion Models
- [Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm): Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction

Awesome concurrent work:
- [Pippo](https://yashkant.github.io/pippo/): High-Resolution Multi-View Humans from a Single Image
- [Avat3r](https://tobias-kirschstein.github.io/avat3r/): Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars

## 📖 Citation

```tex
@inproceedings{taubner2025cap4d,
    author    = {Taubner, Felix and Zhang, Ruihang and Tuli, Mathieu and Lindell, David B.},
    title     = {{CAP4D}: Creating Animatable {4D} Portrait Avatars with Morphable Multi-View Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5318-5330}
}
```

## Acknowledgement
This work was developed in collaboration with and with sponsorship from **LG Electronics**. We gratefully acknowledge their support and contributions throughout the course of this project.
