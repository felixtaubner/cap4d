#!/bin/bash

# Usage message
usage() {
  echo "Usage: $0 [INPUT_VIDEO_PATH] [OUTPUT_PATH]"
  echo
  echo "Notice: PIXEL3DMM_PATH and CAP4D_PATH environment variable must be set!"
  echo
  echo "Arguments:"
  echo "  INPUT_VIDEO_PATH      Path to the input video file or directory of images"
  echo "  OUTPUT_PATH           Path to the output directory"
  echo "  [QUALITY]             Inference quality setting [default|high_quality|medium_quality|low_quality|debug]"
  echo "  [DRIVING_VIDEO_PATH]  Path to the driving video file/directory"
  echo
  echo "Options:"
  echo "  --help             Show this help message and exit"
  exit 1
}

# Handle --help option
if [[ "$1" == "--help" ]]; then
  usage
fi

if [ -z "$PIXEL3DMM_PATH" ]; then
  echo "Error: PIXEL3DMM_PATH environment is not set. Exiting."
  exit 1
fi

if [ -z "$CAP4D_PATH" ]; then
  echo "Error: CAP4D_PATH environment is not set. Exiting."
  exit 1
fi

# Argument checks
if [[ -z "$1" ]]; then
  echo "Error: INPUT_VIDEO_PATH is not set."
  usage
fi

if [[ -z "$2" ]]; then
  echo "Error: OUTPUT_PATH is not set."
  usage
fi

INPUT_VIDEO_PATH=$(realpath "$1")
OUTPUT_PATH=$(realpath "$2")
QUALITY="$3"

if [[ -n "$QUALITY" ]]; then
  echo ""
else
  echo "No quality argument provided. Using default."
  QUALITY="default"
fi

if [[ ! "$QUALITY" =~ ^(default|high_quality|medium_quality|low_quality|debug)$ ]]; then
  echo "Error: Invalid quality level '$QUALITY'"
  echo "Allowed values: default | high_quality | medium_quality | low_quality | debug"
  exit 1
fi

echo "Generating avatar with:"
echo "  Reference video:  $INPUT_VIDEO_PATH"
echo "  Output path:      $OUTPUT_PATH"
echo "  Quality:          $QUALITY"

DRIVING_VIDEO_PATH="$4"

mkdir $OUTPUT_PATH

if [[ -n "$DRIVING_VIDEO_PATH" ]]; then
  echo "Driving video provided at $DRIVING_VIDEO_PATH".
  echo "Begin Pixel3DMM tracking and conversion of driving video."

  DRIVING_VIDEO_TRACKING_PATH=$OUTPUT_PATH/driving_video_tracking/

  bash scripts/track_video_pixel3dmm.sh $DRIVING_VIDEO_PATH $DRIVING_VIDEO_TRACKING_PATH
else
  echo "No driving video provided. "
fi

echo "Running Pixel3DMM tracking and conversion of reference images/video"
REFERENCE_TRACKING_PATH=$OUTPUT_PATH/reference_tracking/
bash scripts/track_video_pixel3dmm.sh $INPUT_VIDEO_PATH $REFERENCE_TRACKING_PATH

echo "Generating images using MMDM"
IMAGE_GEN_PATH=$OUTPUT_PATH/mmdm/
mkdir $IMAGE_GEN_PATH
python cap4d/inference/generate_images.py --config_path configs/generation/$QUALITY.yaml --reference_data_path $REFERENCE_TRACKING_PATH --output_path $IMAGE_GEN_PATH

echo "Fitting GaussianAvatar"
AVATAR_PATH=$OUTPUT_PATH/avatar/
python gaussianavatars/train.py --config_path configs/avatar/$QUALITY.yaml --source_paths $IMAGE_GEN_PATH/reference_images/ $IMAGE_GEN_PATH/generated_images/ --model_path $AVATAR_PATH

if [[ -n "$DRIVING_VIDEO_PATH" ]]; then
  echo "Animating avatar using driving video"
  python gaussianavatars/animate.py --model_path $AVATAR_PATH --target_animation_path $DRIVING_VIDEO_TRACKING_PATH/fit.npz  --target_cam_trajectory_path $DRIVING_VIDEO_TRACKING_PATH/cam_orbit.npz  --output_path $OUTPUT_PATH/rendering_orbit/ --export_ply 1 --compress_ply 0
  python gaussianavatars/animate.py --model_path $AVATAR_PATH --target_animation_path $DRIVING_VIDEO_TRACKING_PATH/fit.npz  --target_cam_trajectory_path $DRIVING_VIDEO_TRACKING_PATH/cam_static.npz  --output_path $OUTPUT_PATH/rendering_static/ --export_ply 1 --compress_ply 0

else
  echo "Animate video with precomputed animations at examples/input/animation/sequence_00/."
  python gaussianavatars/animate.py --model_path $AVATAR_PATH --target_animation_path examples/input/animation/sequence_00/fit.npz  --target_cam_trajectory_path examples/input/animation/sequence_00/orbit.npz  --output_path $OUTPUT_PATH/rendering_orbit/ --export_ply 1 --compress_ply 0
fi
