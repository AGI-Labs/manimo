#! /bin/sh

# Install polymetis from source
cd ..
git clone git@github.com:facebookresearch/fairo.git
cd fairo/polymetis
git reset --hard 0a01a7fa7a7c65b2f9a3aebf5e79040940daf9d2

# This script sets up the environment for manimo. It is meant to be sourced
conda env create -f ./polymetis/environment.yml -n manimo
conda activate manimo
pip install -e ./polymetis

# Build libfranka
# ./scripts/build_libfranka.sh

mkdir -p ./polymetis/build
cd ./polymetis/build

cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=OFF -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF
make -j 2

pip install getch
pip install gym
pip install hydra-core==1.2.0
# Done like this to avoid dependency issues
pip install dm-robotics-moma==0.5.0 --no-deps
pip install dm-robotics-transformations==0.5.0 --no-deps
pip install dm-robotics-agentflow==0.5.0 --no-deps
pip install dm-robotics-geometry==0.5.0 --no-deps
pip install dm-robotics-manipulation==0.5.0 --no-deps
pip install dm-robotics-controllers==0.5.0 --no-deps
pip install mujoco==2.3.2 --no-deps
pip install protobuf==3.20.3
pip install pyrealsense2
pip install h5py
 
# teleop setup
cd ../../../..
git clone git@github.com:rail-berkeley/oculus_reader.git
cd oculus_reader
pip install .
