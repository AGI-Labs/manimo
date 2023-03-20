import argparse
from enum import Enum
import glob
import hydra
from manimo.environments.single_arm_env import SingleArmEnv
from manimo.teleoperation.teleop_agent import TeleopAgent
from manimo.utils.helpers import HOMES
from manimo.utils.logger import DataLogger
import numpy as np
import os
import time

# create a single arm environment
def _create_data_folder(dir, input, task, index):
    data_folder = f"{dir}/{input}_{task}_{index}"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    return data_folder

def _get_filename(dir, input, task):
    index = 0
    for name in glob.glob("{}/{}_{}_*.h5".format(dir, input, task)):
        n = int(name[:-4].split("_")[-1])
        if n >= index:
            index = n + 1
    return "{}/{}_{}_{}.h5".format(dir, input, task, index), index

class ButtonState(Enum):
    OFF = 0,
    INPROGRESS = 1,
    ON = 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="demo", help="name of the demo")
    parser.add_argument("--time", type=float, default=100)
    parser.add_argument("--path", type=str, default="./demos/")
    parser.add_argument("--gripper_en", type=bool, default=True )

    args = parser.parse_args()
    name = args.name
    TIME = args.time
    HZ = 30

    hydra.initialize(
            config_path="../conf", job_name="collect_demos_test"
        )

    actuators_cfg = hydra.compose(config_name="actuators")
    sensors_cfg = hydra.compose(config_name="sensors")
    print(f'sensors_cfg: {sensors_cfg}')

    env = SingleArmEnv(sensors_cfg, actuators_cfg, hz=HZ)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    agent = TeleopAgent()

    obs, info = env.reset()

    fname = f"{name}.h5"
    while True:
        save_path = os.path.join(args.path, fname)
        logger = DataLogger(save_path, save_images=True)

        user_in = "r"
        # while user_in == "r":
        obs, info = env.reset()
            # user_in = input("Ready. Recording {}".format(save_path))

        buttons = {}
        num_obs = 0


        # handle button state logic
        log_state: ButtonState = ButtonState.OFF
        apply_rot_state: ButtonState = ButtonState.OFF

        apply_rot_mask = False
        logging = False



        while True:
        # for _ in range(int(TIME * HZ) - 1):
            arm_action, gripper_action, buttons = agent.get_action(
                obs, apply_pos_mask=False
            )

            if buttons:
                log_toggle = buttons['A']
                rotation_toggle = buttons['B']
            # print(f"got an action from then agent")

                if rotation_toggle and apply_rot_state == ButtonState.OFF:
                    apply_rot_mask = True
                    apply_rot_state = ButtonState.INPROGRESS
                    print('applying rot_mask')

                if rotation_toggle and apply_rot_state == ButtonState.ON:
                    apply_rot_mask = False
                    apply_rot_state = ButtonState.INPROGRESS
                    print('disabling rot_mask')

            if log_toggle and log_state == ButtonState.OFF:
                logging = True
                log_state == ButtonState.INPROGRESS
                print('starting logging')

            if log_toggle and log_state == ButtonState.ON:
                logging = False
                log_state == ButtonState.INPROGRESS
                print('stopping logging')


            if arm_action is not None:
                zero_actions = np.zeros_like(arm_action[:3])
                if apply_rot_mask:
                    arm_action[:3] = zero_actions
                    apply_rot_state = ButtonState.ON
                else:
                    apply_rot_state = ButtonState.OFF

                if args.gripper_en:
                    print(f"using gripper action")
                    action = [arm_action, gripper_action]
                else:
                    print(f"not using gripper action")
                    action = [arm_action]
            else:
                action = None
            obs = env.step(action)[0]

            if logging and action is not None:
                # print(f"action: {action}")

                logger.log(obs)
                num_obs += 1

            if logging:
                log_state = ButtonState.ON
            else:
                log_state = ButtonState.OFF


            
        print(f'trigger logger finish')
        logger.finish()
        print('logger finished')

        print(f'number of observations to be logged: {num_obs}')
        print("created new directory!")

        env.reset()
        env.close()

        break

if __name__ == "__main__":
    print(f"main function called")
    main()