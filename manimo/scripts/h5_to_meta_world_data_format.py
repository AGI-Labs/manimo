import argparse
import h5py
import numpy as np
import imageio.v3 as  iio
import imageio
from pathlib import Path
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import cloudpickle

from scipy.spatial.transform import Rotation as R
from typing import Optional
import gc

def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler('xyz', degrees=degrees)
    return euler
# import manimo quat_to_euler

parser = argparse.ArgumentParser()
parser.add_argument("--replay_file", type=str, required=True)
parser.add_argument("--size", type=int, default=256)

args = parser.parse_args()
replay_file = args.replay_file

hz = 30
obs_keys = ['eef_pos', 'eef_rot', 'eef_gripper_width']
action_keys = ['action', 'eef_gripper_action']
img_key = 'images'
obs_key = 'observations'
action_key = 'actions'

# get the replay_file folder name
replay_folder = Path(replay_file).parent

# iterate over all files in the replay_file folder
# sort iterdir
all_replay_files = sorted(replay_folder.iterdir())
all_traj_data = []
for traj_id, replay_file in enumerate(tqdm(all_replay_files)):
    try:
        with h5py.File(replay_file, "r") as f:
            # f = h5py.File(replay_file, "r")

            # get the observations
            traj_data = {}
            traj_data[obs_key] = None
            traj_data[action_key] = None

            for key in obs_keys:
                # extend observation key with f[key]

                values = f[key][:]
                if key == "eef_gripper_width":
                    # apply np.expand_dims to f[key][:], along axis=1
                    # normalize values to [-0.5, 0.5]
                    values = np.expand_dims(f[key][:], axis=1)

                if traj_data[obs_key] is None:
                    traj_data[obs_key] = values
                else:
                    if key == 'eef_rot':
                        # apply quat_to_euler to f[key][:], along axis=1
                        traj_data[obs_key] = np.append(traj_data[obs_key], quat_to_euler(values), axis=1)
                    else:
                        traj_data[obs_key] = np.append(traj_data[obs_key], values, axis=1)

            for key in action_keys:
                
                values = f[key][:]

                if key == "eef_gripper_action":
                    # apply np.expand_dims to f[key][:], along axis=1
                    # normalize values to [-0.5, 0.5]
                    values = np.expand_dims(f[key][:], axis=1)

                if traj_data[action_key] is None:
                    traj_data[action_key] = values
                else:
                    traj_data[action_key] = np.append(traj_data[action_key], values, axis=1)
                    

            videos = f["videos"]
            traj_len = traj_data[action_key].shape[0]
            traj_data[img_key] = np.zeros((traj_len, len(videos), args.size, args.size, 3), dtype=np.uint8)

            for video_idx, video in enumerate(videos):
                data = videos[video][()].tostring()
                # Free up memory by deleting unused variables
                # del videos[video]

                images = imageio.read(data, format="mp4", size=(args.size, args.size), fps=hz)

                # convert mp4 binary to jpeg images
                # correct size is 256x256

                # save images to traj_data[img_key][video]
                for i, image in enumerate(images):
                    traj_data[img_key][i][video_idx] = image
            traj_data['images'] = traj_data['images'][:, [0, 2]]
            all_traj_data.append(traj_data)
            gc.collect()

    except Exception as e:
        print(f"skipping {replay_file}... because of {e}")
        continue

# save the traj data to a new h5 file
# with h5py.File(replay_folder / 'pick_nsh_220_demos.h5', 'w') as new_f:
#     new_f.create_dataset('pick_nsh_220_demos', data=all_traj_data)

try:
    with open(str(replay_folder) + '/pick_nsh_220_demos.pkl', 'wb') as new_f:
        cloudpickle.dump(all_traj_data, new_f)

except Exception as e:
    print(f"Error during saving: {e}")