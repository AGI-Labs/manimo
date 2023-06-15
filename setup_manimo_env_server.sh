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
./scripts/build_libfranka.sh

mkdir -p ./polymetis/build
cd ./polymetis/build

cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_FRANKA=OFF -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF
make -j 2
