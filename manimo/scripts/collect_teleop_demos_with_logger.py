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
    parser.add_argument("--name")
    parser.add_argument("--task", type=str, default="insertion")
    parser.add_argument("--time", type=float, default=100)
    parser.add_argument("--path", type=str, default="data")
    parser.add_argument("--gripper_en", type=bool, default= False)

    args = parser.parse_args()
    name = args.name
    task = args.task
    TIME = args.time
    HZ = 5

    hydra.initialize(
            config_path="../conf", job_name="collect_demos_test"
        )

    actuators_cfg = hydra.compose(config_name="actuators")
    sensors_cfg = hydra.compose(config_name="sensors")

    env = SingleArmEnv(sensors_cfg, actuators_cfg)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    agent = TeleopAgent()

    obs = env.reset()

    while True:
        data_folder = _create_data_folder(args.path, name, task, index)
        filename, index = _get_filename(args.path, name, task)
              
        logger = DataLogger(filename, save_images=True)

        user_in = "r"
        while user_in == "r":
            obs = env.reset()
            user_in = input("Ready. Recording {}".format(filename))

        step = 0
        for _ in range(int(TIME * HZ) - 1):
            start = time.time()

            action = agent.get_action(obs)
            if action is not None:
                arm_action, gripper_action, buttons = action
                if args.gripper_en:
                    action = [arm_action, gripper_action]
                else:
                    action = [arm_action]
            obs = env.step(action)[0]
            logger.log(obs)

            step += 1
            
            end = time.time() - start
        
        # store the last observation
        env.reset()
        logger.finish()
        index += 1

        print(f"created new directory!")
        break

if __name__ == "__main__":
    print(f"main function called")
    main()