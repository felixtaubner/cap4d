#!/bin/bash

if [ -z "$PIXEL3DMM_PATH" ]; then
  echo "Error: PIXEL3DMM_PATH environment is not set. Exiting."
  exit 1
fi

if [ -d "$PIXEL3DMM_PATH" ]; then
  echo "Error: Directory at PIXEL3DMM_PATH ('$PIXEL3DMM_PATH') already exists. Exiting."
  exit 1
fi

if [ -z "$CAP4D_PATH" ]; then
  echo "Error: CAP4D_PATH environment is not set. Exiting."
  exit 1
fi

# install Pixel3DMM while making it compatible with the environment of CAP4D

# install L2CS
pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main
mkdir data/weights/l2cs/
gdown --id 18S956r4jnHtSeT8z8t3z8AoJZjVnNqPJ -O data/weights/l2cs/

# install RobustVideoMatting
mkdir data/weights/rvm/
wget -O data/weights/rvm/rvm_mobilenetv3.pth https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth

# install Pixel3DMM TODO: Set Pixel3DMM installation path

git clone https://github.com/SimonGiebenhain/pixel3dmm/ $PIXEL3DMM_PATH
cd $PIXEL3DMM_PATH
git checkout 98b5d79f18bc478282494be02f984dc93f5c9fe9

pip install git+https://github.com/NVlabs/nvdiffrast.git

grep -v '^numpy' requirements.txt | pip install -r /dev/stdin

# install missing packages
pip install gdown Cython trimesh
pip install insightface==0.7.3

# from pixel3dmm/.install_preprocessing_pipeline.sh

cd src/pixel3dmm/preprocessing/

# facer repository
git clone https://github.com/FacePerceiver/facer
cd facer
cp ../replacement_code/farl.py facer/face_parsing/farl.py
cp ../replacement_code/facer_transform.py facer/transform.py
pip install -e .
cd ..

# MICA
git clone https://github.com/Zielon/MICA
cd MICA
git checkout af22e7a5810d474bc28a1433db533723d6bd2b07
cp ../replacement_code/install_mica_download_flame.sh install.sh
cp ../replacement_code/mica_demo.py demo.py
cp ../replacement_code/mica.py micalib/models/mica.py
printf "$FLAME_USERNAME\n$FLAME_PWD\n" | ./install.sh
python $CAP4D_PATH/scripts/fixes/fix_flame_pickle.py --pickle_path data/FLAME2023/flame2023_no_jaw.pkl
python $CAP4D_PATH/scripts/fixes/fix_flame_pickle.py --pickle_path data/FLAME2023/flame2023.pkl
python $CAP4D_PATH/scripts/fixes/fix_flame_pickle.py --pickle_path data/FLAME2020/generic_model.pkl
cd ..

# PIPNet
git clone https://github.com/jhb86253817/PIPNet.git
cd PIPNet
git checkout b9eab58816437403a34aa5bc3adeafe5081fd36b
cd FaceBoxesV2/utils
# monkey patch some files for newer numpy version compatibility
# sed -i 's/np.int_t/np.int64_t/' nms/gpu_nms.pyx
# sed -i 's/np.int_t/np.int64_t/' nms/cpu_nms.pyx
cp $CAP4D_PATH/scripts/fixes/pipnet_cpu_nms.pyx nms/cpu_nms.pyx 
#
sh make.sh
cd ../..
mkdir snapshots
mkdir snapshots/WFLW/
mkdir snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/
gdown --id 1nVkaSbxy3NeqblwMTGvLg4nF49cI_99C -O snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth

# download Pixel3DMM weights
cd ../../../../
mkdir pretrained_weights
cd pretrained_weights
gdown --id 1SDV_8_qWTe__rX_8e4Fi-BE3aES0YzJY -O ./uv.ckpt
gdown --id 1KYYlpN-KGrYMVcAOT22NkVQC0UAfycMD -O ./normals.ckpt

cd $PIXEL3DMM_PATH
pip install -e .
