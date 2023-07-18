import os
import pickle

import numpy as np
from robobuf.buffers import ObsWrapper, Transition, ReplayBuffer
from scipy.spatial.transform import Rotation as R


def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler


os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

class DataLogger:
    """This class is used to log observations from the robot environment."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        obs_keys=["eef_pos", "eef_rot", "eef_gripper_width"],
        action_keys=["action", "eef_gripper_action"],
    ):
        self._obs_keys = obs_keys
        self._action_keys = action_keys
        self.replay_buffer = replay_buffer
        self.all_obs = []

    def log(self, obs: dict):
        """Log the current observation.
        Args:
                obs (dict): The current observation.
        """
        # obtain keys containing cam images
        self.all_obs.append(obs)

    def dump_to_file(self, filename):
        # Dump to file
        with open(filename, "wb") as f:
            pickle.dump(self.replay_buffer.to_traj_list(), f)

    def finish(self, traj_idx=0):
        for i in range(len(self.all_obs)):
            cur_raw_obs = self.all_obs[i]
            
            obs = {'state': np.empty(0)}
            
            for key in self._obs_keys:
                if key == "eef_rot":
                    obs['state'] = np.append(obs['state'], quat_to_euler(cur_raw_obs[key]))
                else:
                    obs['state'] =  np.append(obs['state'], cur_raw_obs[key])
            
            cam_keys = [key for key in cur_raw_obs.keys() if "cam" in key]
            for key in cam_keys:
                img, ts = cur_raw_obs[key]
                obs[key] = img

            obswrapper = ObsWrapper(obs)

            action = np.empty(0)
            for key in self._action_keys:
                if key in cur_raw_obs.keys():
                    action = np.append(action, cur_raw_obs[key])
            
            reward = 0 if i < len(self.all_obs)-1 else 1

            transition = Transition(obswrapper, action, reward)
            self.replay_buffer.add(transition, is_first=(i==0))

        # get the latest traj_idx from folder ./demos
        if os.path.exists("./demos"):
            latest_traj_idx = len(os.listdir("./demos"))

        self.dump_to_file(f"./demos/traj_{latest_traj_idx:05d}.pkl")
        self.replay_buffer.clear()
        self.all_obs.clear()