import argparse
import hydra
from manimo.environments.single_arm_env import SingleArmEnv
from manimo.teleoperation.teleop_agent import TeleopAgent
from manimo.utils.helpers import HOMES, StateManager
from manimo.utils.logger import DataLogger
import os
import time

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="demo", help="name of the demo")
    parser.add_argument("--time", type=float, default=100)
    parser.add_argument("--path", type=str, default="./demos/")
    parser.add_argument("--gripper_en", type=bool, default=True )

    args = parser.parse_args()
    name = args.name

    hydra.initialize(
            config_path="../conf", job_name="collect_demos_test"
        )

    actuators_cfg = hydra.compose(config_name="actuators")
    sensors_cfg = hydra.compose(config_name="sensors")
    env_cfg = hydra.compose(config_name="env")

    env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    agent = TeleopAgent()

    obs, _ = env.reset()

    # fetch step from final demo value in demos/demo_x.h5
    demo_files = [f for f in os.listdir(f"{args.path}") if f.endswith('.h5')]
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
        save_path = os.path.join(args.path, fname)
        logger = DataLogger(save_path, save_images=True)

        obs, _ = env.reset()
        buttons = {}

        button_state_manager = StateManager()
        logging, finish_logging = False, False
        num_obs, start_time = 0, time.time()

        print(f"ready to collect demos with name: {fname}!")
        while True:
            arm_action, gripper_action, buttons = agent.get_action(
                obs, apply_pos_mask=False
            )

            # Handle the 'A' button state and toggle the logging state
            button_state_manager.handle_state(
                buttons,
                'A',
                lambda: (setattr((logging, num_obs, start_time, finish_logging), *toggle_logger(logging, logger, num_obs, start_time)))
            )

            if arm_action is not None:
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

            if finish_logging:
                break
        step += 1

    env.reset()
    env.close()
    
        # break

if __name__ == "__main__":
    print(f"main function called")
    main()