import argparse
import hydra
from manimo.environments.single_arm_env import SingleArmEnv
from manimo.teleoperation.teleop_agent import TeleopAgent
from manimo.utils.helpers import HOMES, StateManager
from manimo.utils.logger import DataLogger
from pathlib import Path
import os
import torch
import numpy as np
import cv2
import time
import yaml
from scipy.spatial.transform import Rotation as R

def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler('xyz', degrees=degrees)
    return euler


def toggle_logger(logging, logger, num_obs, start_time):
    finish_logging = False
    if logging:
        effective_hz = num_obs / (time.time() - start_time)
        logger.finish()
        finish_logging = True
        print(f"logger closed, with framerate: {effective_hz}")
    else:
        print(f"logger started!")
        start_time = time.time()
        num_obs = 0

    logging = not logging
    return logging, num_obs, start_time, finish_logging

def toggle_agent(use_ai_agent):
    return not use_ai_agent


class AIAgent:
    def __init__(self, agent_path):
        with open(Path(agent_path, 'agent_config.yaml'), 'r') as f:
            config_yaml = f.read()
            agent_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, 'obs_config.yaml'), 'r') as f:
            config_yaml = f.read()
            obs_config = yaml.safe_load(config_yaml)

        agent = hydra.utils.instantiate(agent_config)
        agent.load_state_dict(torch.load(Path(agent_path, 'r3m_stacking_newdata2.ckpt'), map_location='cpu')['model'])
        self.agent = agent.eval().cuda()
        self.actions = []

        self.transform = hydra.utils.instantiate(obs_config['transform'])
        self.img_keys = obs_config["img"]
        self.obs_keys = obs_config["obs"]

        print(f"loaded agent from {agent_path}")

    def get_raw_imgs_and_obs(self, env_obs, IMG_SIZE=256):

        imgs = [env_obs[cam_key][0] for cam_key in self.img_keys]
        # append obs to raw_obs
        raw_obs = []
        for obs_key in self.obs_keys:
            if obs_key == 'eef_rot':
                raw_obs.extend(quat_to_euler(env_obs[obs_key]))
            else:
                if obs_key == 'eef_gripper_width':
                    raw_obs.extend([env_obs[obs_key]])
                else:
                    raw_obs.extend(env_obs[obs_key])

        resized_imgs = [cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA) for img in imgs]
        resized_imgs = np.array(resized_imgs, dtype=np.uint8)

        return resized_imgs, np.array(raw_obs)
    

    def get_action(self, obs):
        if len(self.actions) == 0:
            raw_imgs, raw_obs = self.get_raw_imgs_and_obs(obs)
            obs = torch.from_numpy(raw_obs).float()[None].cuda()
            img = self.transform(torch.from_numpy(raw_imgs).float().permute((0, 3, 1, 2)) / 255)[None].cuda()
            with torch.no_grad():
                acs = self.agent.eval().get_actions(img, obs)

            acs = acs.cpu().numpy()[0]

            if len(acs.shape) == 1:
                acs = acs[None]
            
            self.actions = acs
        
        new_action = self.actions[0]
        self.actions = self.actions[1:]

        arm_action = new_action[:6]
        gripper_action = float(new_action[6] > 0.8)
        
        return [arm_action, gripper_action]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="demo", help="name of the demo")
    parser.add_argument("--data_path", type=str, default="./demos/")
    parser.add_argument("--agent_paths", action="append", default=[])

    args = parser.parse_args()
    name = args.name

    # create a list of ai agents based on the agent paths, use list comprehension
    ai_agents = [AIAgent(agent_path) for agent_path in args.agent_paths]

    hydra.initialize(
            config_path="../conf", job_name="collect_demos"
        )

    actuators_cfg = hydra.compose(config_name="actuators")
    sensors_cfg = hydra.compose(config_name="sensors")
    env_cfg = hydra.compose(config_name="env")

    env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    

    teleop_agent = TeleopAgent()

    obs, _ = env.reset()

    # fetch step from final demo value in demos/demo_x.h5
    demo_files = [f for f in os.listdir(f"{args.data_path}") if f.endswith('.h5')]
    demo_files.sort()

    # take step from last demo file
    if len(demo_files):
        demo_file = demo_files[-1]
        step = int(demo_file.split('.')[0].split('_')[1])
    else:
        step = -1

    step += 1

    while True:
        fname = f"{name}_{step:04d}.h5"
        save_path = os.path.join(args.data_path, fname)
        logger = DataLogger(save_path, save_images=True)

        obs, _ = env.reset()
        buttons = {}

        button_state_manager = StateManager()
        logging, finish_logging = False, False
        num_obs, start_time = 0, time.time()
        use_ai_agent = False
        agent_idx = 0

        print(f"ready to collect demos with name: {fname}!")
        while True:
            arm_action, gripper_action, buttons = teleop_agent.get_action(
                obs)

            # Handle the 'A' button state and toggle the logging state
            result = button_state_manager.handle_state(
                buttons,
                'A',
                toggle_logger, logging, logger, num_obs, start_time
            )
            if result:
                logging, num_obs, start_time, finish_logging = result

            # Handle the 'B' button state and toggle the use_ai_agent state
            result = button_state_manager.handle_state(
                buttons,
                'B',
                toggle_agent, use_ai_agent
            )
            if result is not None:
                use_ai_agent = result        

            if arm_action is not None:
                teleop_action = [arm_action, not gripper_action]
            else:
                teleop_action = None

            if use_ai_agent:
                action = ai_agents[agent_idx].get_action(obs)
                print(f"using ai agent {agent_idx}")
            else:
                action = teleop_action
                print(f"using teleop agent")
            
            obs = env.step(action)[0]

            if logging and action is not None:
                logger.log(obs)
                num_obs += 1

            if finish_logging:
                break

        step += 1



if __name__ == "__main__":
    print(f"main function called")
    main()