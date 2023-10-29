from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import yaml
from dataloaders.utils import get_transform_by_name
from manimo.environments.single_arm_env import SingleArmEnv
from scipy.spatial.transform import Rotation as R


def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler


class AIAgent:
    def __init__(self, base_path):
        with open(Path(base_path, "agent_config.yaml"), "r") as f:
            config_yaml = f.read()
            agent_config = yaml.safe_load(config_yaml)
        print("Constructing agent...")
        print(config_yaml)

        agent = hydra.utils.instantiate(agent_config)
        agent.load_state_dict(
            torch.load(
                Path(base_path, "recheck_stacking.ckpt"), map_location="cpu"
            )["model"]
        )
        self.agent = agent.eval().cuda()
        self.transform = get_transform_by_name("preproc")
        print(f"Agent loaded from {base_path}")

    def get_action(self, raw_imgs, raw_obs):
        obs = torch.from_numpy(raw_obs).float()[None].cuda()
        img = self.transform(
            torch.from_numpy(raw_imgs).float().permute((0, 3, 1, 2)) / 255
        )[None].cuda()
        start = time.time()
        with torch.no_grad():
            acs = self.agent.eval().get_actions(img, obs)
        print(1.0 / (time.time() - start))
        return acs.cpu().numpy()[0]


def get_raw_imgs_and_obs(env_obs, IMG_SIZE=256):
    cam_keys = ["cam1_left"]
    obs_keys = ["eef_pos", "eef_rot", "eef_gripper_width"]

    imgs = [env_obs[cam_key][0] for cam_key in cam_keys]
    # append obs to raw_obs
    raw_obs = []
    for obs_key in obs_keys:
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
    cv2.imshow("test", resized_imgs[-1][:, :, ::-1])
    cv2.waitKey(1)
    resized_imgs = np.array(resized_imgs, dtype=np.uint8)

    return resized_imgs, np.array(raw_obs)


def main():
    base_path = "/home/sudeep/Downloads/recheck_stacking/franka_r3m_2023-05-11_02-16-45"
    # base_path = '/home/sudeep/Downloads/recheck_stacking/franka_vit_base_2023-05-11_02-17-31'

    agent = AIAgent(base_path)

    hydra.initialize(config_path="../conf", job_name="policy_inference")

    # create the environment
    actuators_cfg = hydra.compose(config_name="actuators")
    sensors_cfg = hydra.compose(config_name="sensors")
    env_cfg = hydra.compose(config_name="env")

    env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg)

    MAX_STEPS = 300
    while True:
        obs, info = env.reset()
        raw_imgs, raw_obs = get_raw_imgs_and_obs(obs)

        # wait for enter key to continue
        print("Press enter to continue")
        input()

        step = 0

        while step < MAX_STEPS:
            # action can be a list, use the list of actions for future
            # len(list) steps before calling get action again
            with torch.no_grad():
                acs = agent.get_action(raw_imgs, raw_obs)

            if len(acs.shape) == 1:
                acs = [acs]
            for ac in acs:
                arm_action = ac[:6]
                gripper_action = float(ac[6] > 0.8)
                obs, info, _, _ = env.step([arm_action, gripper_action])
                print(f"Step: {step}, Action: {ac}")
                raw_imgs, raw_obs = get_raw_imgs_and_obs(obs)
                step += 1
                if step >= MAX_STEPS:
                    break
        print(f"done, reseting env")
        env.reset()


if __name__ == "__main__":
    main()
