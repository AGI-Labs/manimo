# #! /bin/sh

# Install polymetis from source
cd ..
# clone & create *gpu* env
git clone git@github.com:hengyuan-hu/monometis.git
cd monometis/
mamba env create -f polymetis/environment.yml -n manimo-latest
conda activate manimo-latest
pip install mkl
# compile stuff, no need to build libfranka on this machine
mkdir -p ./polymetis/build
cd ./polymetis/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ../..

pip install -e ./polymetis

# install robobuf
pip install git+https://github.com/AGI-Labs/robobuf

# ik setup
pip install getch
pip install gym
pip install hydra-core==1.2.0

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
pip install opencv-python
pip install pytimedinput
pip install dm-control==1.0.5
pip install lxml

# teleop setup
cd ../
git clone git@github.com:rail-berkeley/oculus_reader.git
cd oculus_reader
pip install .
