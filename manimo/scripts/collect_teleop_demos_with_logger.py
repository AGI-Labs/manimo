import argparse
import os
import time
from enum import Enum

import hydra
import numpy as np
from manimo.environments.single_arm_env import SingleArmEnv
from manimo.teleoperation.teleop_agent import TeleopAgent
from manimo.utils.logger import DataLogger


class ButtonState(Enum):
    OFF = (0,)
    INPROGRESS = (1,)
    ON = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default="demo", help="name of the demo"
    )
    parser.add_argument("--time", type=float, default=100)
    parser.add_argument("--path", type=str, default="./demos/")
    parser.add_argument("--gripper_en", type=bool, default=True)

    args = parser.parse_args()
    name = args.name

    hydra.initialize(config_path="../conf", job_name="collect_demos_test")

    actuators_cfg = hydra.compose(config_name="actuators")
    sensors_cfg = hydra.compose(config_name="sensors")
    env_cfg = hydra.compose(config_name="env")

    env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    agent = TeleopAgent()
    print(f"loaded teleop agent: {agent}")

    obs, info = env.reset()

    # fetch step from final demo value in demos/demo_x.h5
    demo_files = [f for f in os.listdir("./demos") if f.endswith(".h5")]
    demo_files.sort()

    # take step from last demo file
    if len(demo_files):
        demo_file = demo_files[-1]
        step = int(demo_file.split(".")[0].split("_")[1])
    else:
        step = -1
    step += 1
    while True:
        fname = f"{name}_{step:04d}.h5"
        save_path = os.path.join(args.path, fname)
        logger = DataLogger(save_path, save_images=True)

        obs, _ = env.reset()

        buttons = {}
        # handle button state logic
        log_state: ButtonState = ButtonState.OFF

        apply_rot_mask = False
        logging = False
        log_toggle = False
        apply_rot_mask = False
        print(f"ready to collect demos with name: {fname}!")
        while True:
            arm_action, gripper_action, buttons = agent.get_action(obs)

            if buttons:
                log_toggle = buttons["A"]

                if log_toggle:
                    log_state = ButtonState.INPROGRESS

                if not log_toggle and log_state == ButtonState.INPROGRESS:
                    logging = not logging
                    log_state = ButtonState.OFF

                    if not logging:
                        effective_hz = num_obs / (time.time() - start)
                        logger.finish()
                        print(f"logger closed, with framerate: {effective_hz}")
                        break
                    else:
                        print(f"logger started!")
                        num_obs, start = 0, time.time()

            if arm_action is not None:
                zero_actions = np.zeros_like(arm_action[:3])
                if apply_rot_mask:
                    arm_action[:3] = zero_actions

                if args.gripper_en:
                    action = [arm_action, not gripper_action]
                else:
                    action = [arm_action]
            else:
                action = None

            obs = env.step(action)[0]

            if logging and action is not None:
                logger.log(obs)
                num_obs += 1

        step += 1


if __name__ == "__main__":
    print(f"main function called")
    main()
