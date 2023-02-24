import argparse
import glob
import hydra
from manimo.environments.single_arm_env import SingleArmEnv
from manimo.teleoperation.teleop_agent import TeleopAgent
from manimo.utils.helpers import HOMES
from manimo.utils.logger import DataLogger
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="demo", help="name of the demo")
    parser.add_argument("--time", type=float, default=100)
    parser.add_argument("--path", type=str, default="./demos/")
    parser.add_argument("--gripper_en", type=bool, default=False)

    args = parser.parse_args()
    name = args.name
    TIME = args.time
    HZ = 20

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
        while user_in == "r":
            obs, info = env.reset()
            user_in = input("Ready. Recording {}".format(save_path))

        buttons = {}
        start_logging = False
        stop_logging = False
        apply_pos_mask = False
        logging = False
        num_obs = 0
        for _ in range(int(TIME * HZ) - 1):
            arm_action, gripper_action, buttons = agent.get_action(
                obs, apply_pos_mask=apply_pos_mask
            )
            if arm_action is not None:
                if args.gripper_en:
                    action = [arm_action, gripper_action]
                else:
                    action = [arm_action]
            else:
                action = None
            obs = env.step(action)[0]

            if logging and action is not None:
                print(f"action: {action}")

                logger.log(obs)
                num_obs += 1

            if buttons:
                start_logging = buttons['A']
                stop_logging = buttons['B']
                apply_pos_mask = buttons['RTr']
            
            if start_logging and not logging:
                logging = True
                print('starting logging')

            if stop_logging and logging:
                logging = False
                print('stopping logging')
                break
            
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