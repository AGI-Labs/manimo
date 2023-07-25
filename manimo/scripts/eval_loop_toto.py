import argparse
from pathlib import Path
import time
import cv2
import hydra
import numpy as np
import torch
import yaml
from manimo.scripts.manimo_loop import ManimoLoop

from manimo.utils.callbacks import BaseCallback
from pytimedinput import timedKey
from scipy.spatial.transform import Rotation as R
from manimo.utils.new_logger import DataLogger
from robobuf.buffers import ReplayBuffer
torch.set_float32_matmul_precision('high')
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import py_tf_eager_policy
import tf_agents

def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler

'''
class AIAgent:
    def __init__(self, agent_path, model_name="r3m_stacking_newdata2.ckpt"):
        with open(Path(agent_path, "agent_config.yaml"), "r") as f:
            config_yaml = f.read()
            agent_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "obs_config.yaml"), "r") as f:
            config_yaml = f.read()
            obs_config = yaml.safe_load(config_yaml)

        agent = hydra.utils.instantiate(agent_config)
        agent.load_state_dict(
            torch.load(Path(agent_path, model_name), map_location="cpu")[
                "model"
            ]
        )
        self.agent = torch.compile(agent.eval().cuda())
        # self.agent = agent.eval().cuda()
        self.actions = []

        self.transform = hydra.utils.instantiate(obs_config["transform"])
        self.img_keys = obs_config["img"]
        self.obs_keys = obs_config["obs"]

        print(f"loaded agent from {agent_path}")

    def get_raw_imgs_and_obs(self, env_obs, IMG_SIZE=256):
        imgs = [env_obs[cam_key][0] for cam_key in self.img_keys]
        # append obs to raw_obs
        raw_obs = []
        for obs_key in self.obs_keys:
            if obs_key == "eef_rot":
                raw_obs.extend(quat_to_euler(env_obs[obs_key]))
            else:
                if obs_key == "eef_gripper_width":
                    raw_obs.extend([env_obs[obs_key]])
                else:
                    raw_obs.extend(env_obs[obs_key])

        resized_imgs = [
            cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            for img in imgs
        ]
        resized_imgs = np.array(resized_imgs, dtype=np.uint8)

        return resized_imgs, np.array(raw_obs)

    def get_action(self, obs):
        if len(self.actions) == 0:
            raw_imgs, raw_obs = self.get_raw_imgs_and_obs(obs)
            obs = torch.from_numpy(raw_obs).float()[None].cuda()
            img = self.transform(
                torch.from_numpy(raw_imgs).float().permute((0, 3, 1, 2)) / 255
            )[None].cuda()
            start = time.time()
            with torch.no_grad():
                acs = self.agent.eval().get_actions(img, obs)
            print(f"get action time: {(time.time() - start)*1000} ms")
            acs = acs.cpu().numpy()[0]

            if len(acs.shape) == 1:
                acs = acs[None]

            self.actions = acs

        new_action = self.actions[0]
        self.actions = self.actions[1:]

        arm_action = new_action[:6]
        gripper_action = float(new_action[6] > 0.8)

        return [arm_action, gripper_action]
'''

class AIAgent:
    def __init__(self, agent_path, scale_actions=True):
        self.tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
                model_path=agent_path,
                load_specs_from_pbtxt=True,
                use_tf_function=True)
        embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
        task = 'pour'

        def normalize_task_name(task_name):
            replaced = task_name.replace('_', ' ').replace('1f', ' ').replace(
                                        '4f', ' ').replace('-', ' ').replace('50',
                                        ' ').replace('55',' ').replace('56', ' ')
            return replaced.lstrip(' ').rstrip(' ')

        self.task_embedding = embed([normalize_task_name(task)])[0]
        self.policy_state = self.tfa_policy.get_initial_state(batch_size=1)
        self.img_keys = ['cam0c']

        if scale_actions:
            self.scale_factor = 0.1
        else:
            self.scale_factor = 1

        self.action_space = 1

        print(f"loaded agent from {agent_path}, set scale_factor: {self.scale_factor}!")

    def get_raw_imgs_and_obs(self, env_obs, IMG_SIZE_H=256, IMG_SIZE_W=320):

        imgs = [env_obs[cam_key][0] for cam_key in self.img_keys]
        # append obs to raw_obs

        resized_imgs = [tf.image.resize_with_pad(img, target_width=320, target_height=256) for img in imgs]
        resized_imgs = [tf.cast(resized_img, np.uint8) for resized_img in resized_imgs]

        return resized_imgs, None
    

    def get_action(self, obs):
        start = time.time()
        raw_imgs, _ = self.get_raw_imgs_and_obs(obs)
        observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(self.tfa_policy.time_step_spec.observation))
        observation['image'] = raw_imgs[0].numpy()
        observation['natural_language_embedding'] = self.task_embedding

        tfa_time_step = ts.transition(observation, reward=np.zeros(()))
        policy_step = self.tfa_policy.action(tfa_time_step, self.policy_state)
        self.policy_state = policy_step.state
        
        action = policy_step.action

        world_vector = action['world_vector']/self.scale_factor
        rotation_delta = action['rotation_delta']/self.scale_factor

        new_action = np.concatenate([world_vector, rotation_delta])
        
        end = time.time()
        print(f"time taken to compute pred: {end-start} seconds")
        return [new_action]


class Eval(BaseCallback):
    """
    Teleoperation callback.
    """

    def __init__(self, logger, agent_paths):
        super().__init__(logger)
        self.logger = logger
        self.ai_agents = [
            AIAgent(Path(agent_path), scale_actions=False)
            for agent_path in agent_paths
        ]
        print(f"loaded all ai agents!")
        self.agent_paths = agent_paths
        self.agent_idx = 0

    def on_begin_traj(self, traj_idx):
        """
        Called at the beginning of a trajectory.
        """
        # allow user to select agent
        print(
            f"current agent_idx: {self.agent_idx} at path:"
            f" {self.agent_paths[self.agent_idx]}"
        )
        print(
            "press enter to use the same agent, press any other number to"
            f" chose agent index from 0 to {len(self.ai_agents) - 1}"
        )
        print(f"to use agents at paths: {self.agent_paths}")
        user_input = input()
        if user_input != "":
            agent_idx = int(user_input)
            if agent_idx < 0 or agent_idx >= len(self.ai_agents):
                print(
                    f"invalid agent_idx: {self.agent_idx}, using agent_idx: 0"
                )
                agent_idx = 0
            self.agent_idx = agent_idx

    def on_end_traj(self, traj_idx):
        print(f"finish logging")
        # self.logger.finish()
        pass
        # return super().on_end_traj(traj_idx)

    def get_action(self, obs):
        action = self.ai_agents[self.agent_idx].get_action(obs)
        new_obs = obs.copy()
        new_obs['action'] = action[0]#np.append(*action)
        # self.logger.log(new_obs)
        return action

    def on_step(self, traj_idx, step_idx):
        start_time = time.time()
        userText, timedOut = timedKey(
            "reset if r, otherwise continue",
            allowCharacters="r",
            timeout=0.001,
        )
        # print(f"took {time.time() - start_time} to get user input")
        if not timedOut:
            print(f"pressed {userText}, resetting")
            return True
        else:
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_paths", nargs="+", default=[])
    args = parser.parse_args()

    # replay_buffer = ReplayBuffer(max_size=1000)
    # logger = DataLogger(replay_buffer=replay_buffer, action_keys=["action"])

    eval_callback = Eval(None, args.agent_paths)
    manimo_loop = ManimoLoop(callbacks=[eval_callback])

    manimo_loop.run()


if __name__ == "__main__":
    main()
