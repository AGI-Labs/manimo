import argparse

import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--replay_file", type=str, required=True)

args = parser.parse_args()
replay_file = args.replay_file

hz = 30

# read the test.h5 file
f = h5py.File(replay_file, "r")

videos = f["videos"]

for video in videos:
    data = videos[video][()].tostring()
    with open(f"{video}.mp4", "wb") as f:
        f.write(data)
