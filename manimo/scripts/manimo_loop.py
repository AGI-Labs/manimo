import argparse
import os
import time
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import yaml
from manimo.environments.single_arm_env import SingleArmEnv
from manimo.teleoperation.teleop_agent import TeleopAgent
from manimo.utils.helpers import StateManager
from manimo.utils.logger import DataLogger
from pytimedinput import timedKey

from scipy.spatial.transform import Rotation as R


def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler


def toggle_logger(logging, logger, num_obs, start_time):
    finish = False
    if logging:
        effective_hz = num_obs / (time.time() - start_time)
        logger.finish()
        finish = True
        print(f"logger closed, with framerate: {effective_hz}")
    else:
        print("logger started!")
        start_time = time.time()
        num_obs = 0

    logging = not logging
    return logging, num_obs, start_time, finish

def toggle(flag):
    return not flag


class AIAgent:
    def __init__(self, agent_path, model_name='r3m_stacking_newdata2.ckpt'):
        with open(Path(agent_path, 'agent_config.yaml'), 'r') as f:
            config_yaml = f.read()
            agent_config = yaml.safe_load(config_yaml)
        with open(Path(agent_path, "obs_config.yaml"), "r") as f:
            config_yaml = f.read()
            obs_config = yaml.safe_load(config_yaml)

        agent = hydra.utils.instantiate(agent_config)
        agent.load_state_dict(torch.load(Path(agent_path, model_name), map_location='cpu')['model'])
        self.agent = agent.eval().cuda()
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
    parser.add_argument(
        "--name", type=str, default="demo", help="name of the demo"
    )
    parser.add_argument("--data_path", type=str, default="./demos/")
    parser.add_argument("--agent_paths",  nargs='+', default=[])
    parser.add_argument("--enable_teleop", action='store_true', help="enable teleop")
    parser.add_argument("--enable_dagger", action='store_true', help="enable dagger")
    parser.add_argument("--save_demos", action='store_true', help="save demos")
    parser.add_argument("--T", type=int, default=280, help="Number of timesteps to collect")

    args = parser.parse_args()
    name = args.name

    print(f"using time horizon: {args.T}")

    # create a list of ai agents based on the agent paths, use list comprehension
    ai_agents = [AIAgent(Path(agent_path).parent, Path(agent_path).name) for agent_path in args.agent_paths]

    hydra.initialize(config_path="../conf", job_name="collect_demos")

    actuators_cfg = hydra.compose(config_name="actuators")
    sensors_cfg = hydra.compose(config_name="sensors")
    env_cfg = hydra.compose(config_name="env")

    env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    if args.enable_teleop:
        teleop_agent = TeleopAgent()

    obs, _ = env.reset()

    # fetch step from final demo value in demos/demo_x.h5
    demo_files = [
        f for f in os.listdir(f"{args.data_path}") if f.endswith(".h5")
    ]
    demo_files.sort()

    # take step from last demo file
    if len(demo_files):
        demo_file = demo_files[-1]
        step = int(demo_file.split(".")[0].split("_")[1])
    else:
        step = -1

    step += 1
    agent_idx = 0
    while True:
        if args.save_demos:
            fname = f"{name}_{step:04d}.h5"
            save_path = os.path.join(args.data_path, fname)
            logger = DataLogger(save_path, save_images=True)
            print(f"ready to collect demos with name: {fname}!")

        obs, _ = env.reset()
        # sleep for 1 seconds
        time.sleep(1)
        buttons = {}

        button_state_manager = StateManager()
        logging, finish = False, False
        num_obs, start_time = 0, time.time()
        use_ai_agent = 1#not args.enable_teleop

        env_steps = 0

        if args.agent_paths:
            print(f"current agent_idx: {agent_idx} at path: {args.agent_paths[agent_idx]}")
            print(f"press enter to use the same agent, press any other number to chose agent index from 0 to {len(ai_agents) - 1}")
            print(f"to use agents at paths: {args.agent_paths}")
        user_input = input()
        if user_input != "":
            agent_idx = int(user_input)
            if agent_idx < 0 or agent_idx >= len(ai_agents):
                print(f"invalid agent_idx: {agent_idx}, using agent_idx: 0")
                agent_idx = 0

        print(f"using agent_idx: {agent_idx}")
        # wait for 3 seconds
        # time.sleep(2)
        # while True:
        while env_steps < args.T:
            # if user presses ctrl c then break

            if args.enable_teleop:
                arm_action, gripper_action, buttons = teleop_agent.get_action(
                    obs
                )
                if arm_action is not None:
                    teleop_action = [arm_action, not gripper_action]
                else:
                    teleop_action = None

            # Handle the 'A' button state and toggle the logging state
            if args.save_demos:
                result = button_state_manager.handle_state(
                    buttons,
                    'A',
                    toggle_logger, logging, logger, num_obs, start_time
                )
                if result:
                    logging, num_obs, start_time, finish = result
            else:
                result = button_state_manager.handle_state(
                    buttons,
                    'A',
                    toggle, finish
                )
                if result is not None:
                    finish = result

            # Handle the 'B' button state and toggle the use_ai_agent state
            if args.enable_dagger:
                result = button_state_manager.handle_state(
                    buttons,
                    'B',
                    toggle, use_ai_agent
                )
                if result is not None:
                    use_ai_agent = result
            else:
                result = button_state_manager.handle_state(
                    buttons,
                    'B',
                    toggle, finish
                )
                if result is not None:
                    finish = result

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
            if finish:
                # get agent_idx from user keyboard input, if the user presses enter, then use the same agent
                # otherwise, use the next agent
                break

            env_steps += 1

            userText, timedOut = timedKey("reset if r, otherwise continue", allowCharacters="r", timeout=0.025)
            if not timedOut:
                print(f"pressed {userText}, resetting")
                break

        step += 1


if __name__ == "__main__":
    print("main function called")
    main()
