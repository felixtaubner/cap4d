#!/bin/bash

# Run quick inference to test pipeline installation
mkdir examples/output

# Test MMDM installation by generating a few images
python cap4d/inference/generate_images.py \
    --config_path configs/generation/high_quality.yaml \
    --reference_data_path examples/input/tesla/ \
    --output_path examples/output/tesla/

# Test GaussianAvatars installation by fitting for a few iterations
python gaussianavatars/train.py \
    --config_path configs/avatar/default.yaml \
    --source_paths examples/output/tesla/reference_images/ examples/output/tesla/generated_images/ \
    --model_path examples/output/tesla/avatar/

# Test rendering and export 
python gaussianavatars/animate.py \
    --model_path examples/output/tesla/avatar/ \
    --target_animation_path examples/input/animation/sequence_00/fit.npz \
    --target_cam_trajectory_path examples/input/animation/sequence_00/orbit.npz  \
    --output_path examples/output/tesla/animation_00/ \
    --export_ply 1 \
    --compress_ply 0
