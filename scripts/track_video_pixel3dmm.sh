#!/bin/bash

# Usage message
usage() {
  echo "Usage: $0 [INPUT_VIDEO_PATH] [OUTPUT_PATH]"
  echo
  echo "Notice: PIXEL3DMM_PATH and CAP4D_PATH environment variable must be set!"
  echo
  echo "Arguments:"
  echo "  INPUT_VIDEO_PATH   Path to the input video file or directory of images"
  echo "  OUTPUT_PATH        Path to the output directory"
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

echo "Processing video:"
echo "  Input:  $INPUT_VIDEO_PATH"
echo "  Output: $OUTPUT_PATH"

if [ -e "$INPUT_VIDEO_PATH" ]; then
  echo "Found input video ('$INPUT_VIDEO_PATH')."
else
  echo "Error: No file or directory found at ('$INPUT_VIDEO_PATH'). Exiting."
  exit 1
fi

IS_DISCONTINOUS=True
if [ -f "$INPUT_VIDEO_PATH" ]; then
  IS_DISCONTINOUS=False
fi

mkdir $OUTPUT_PATH

echo "Running Pixel3DMM script"

# set required environment variables
export PIXEL3DMM_CODE_BASE=$PIXEL3DMM_PATH
export PIXEL3DMM_PREPROCESSED_DATA=$OUTPUT_PATH/pixel3dmm_preprocessing/
export PIXEL3DMM_TRACKING_OUTPUT=$OUTPUT_PATH/pixel3dmm_tracking/

export PATH_TO_VIDEO=$INPUT_VIDEO_PATH
base_name=$(basename $PATH_TO_VIDEO)
export VID_NAME="${base_name%%.*}"

# switch to PIXEL3DMM_PATH
cd $PIXEL3DMM_PATH

# run preprocessing
python scripts/run_preprocessing.py --video_or_images_path $PATH_TO_VIDEO

# run UV and normal prediction
python scripts/network_inference.py model.prediction_type=normals video_name=$VID_NAME
python scripts/network_inference.py model.prediction_type=uv_map video_name=$VID_NAME

# run
python scripts/track.py video_name=$VID_NAME include_neck=False use_flame2023=True ignore_mica=True is_discontinuous=$IS_DISCONTINOUS

echo "Converting Pixel3DMM tracking to FlowFace"

# switch to CAP4D_PATH
cd $CAP4D_PATH

TRACKING_DIR=$(find "$PIXEL3DMM_TRACKING_OUTPUT" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)

# run conversion to FlowFace
python scripts/pixel3dmm/convert_to_flowface.py --video_path $INPUT_VIDEO_PATH --tracking_path $TRACKING_DIR --preprocess_path $PIXEL3DMM_PREPROCESSED_DATA/$VID_NAME --output_path $OUTPUT_PATH
