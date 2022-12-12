import argparse
import glob
import hydra
from manimo.environments.single_arm_env import SingleArmEnv
from manimo.teleoperation.teleop_agent import TeleopAgent
from manimo.utils.helpers import HOMES
import numpy as np
import os

# create a single arm environment
def _get_filename(dir, input, task):
    index = 0
    for name in glob.glob("{}/{}_{}_*.npz".format(dir, input, task)):
        n = int(name[:-4].split("_")[-1])
        if n >= index:
            index = n + 1
    return "{}/{}_{}_{}.npz".format(dir, input, task, index)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--task", type=str, default="insertion")
    parser.add_argument("--time", type=float, default=100)

    args = parser.parse_args()
    name = args.name
    task = args.task
    TIME = args.time
    HZ = 30
    home = HOMES[task]

    hydra.initialize(
            config_path="../conf", job_name="collect_demos_test"
        )

    actuators_cfg = hydra.compose(config_name="actuators_record")
    sensors_cfg = hydra.compose(config_name="sensors")

    env = SingleArmEnv(sensors_cfg, actuators_cfg)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    agent = TeleopAgent()

    obs = env.reset()

    while True:
        filename = _get_filename("data", name, task)

        user_in = "r"
        while user_in == "r":
            obs = env.reset()
            user_in = input("Ready. Recording {}".format(filename))

        joints = []
        eef_positions = []
        eef_orientations = []
        for _ in range(int(TIME * HZ) - 1):

            action = agent.get_action(obs)
            if action is not None:
                action = [action]
            
            obs = env.step(action)[0]
            
            # store observations
            joints.append(obs["q_pos"])
            eef_positions.append(obs["eef_pos"])
            eef_orientations.append(obs["eef_rot"])

        env.reset()

        if not os.path.exists("./data"):
            os.mkdir("data")

        print(f"created new directory!")
        np.savez(filename, home=home, hz=HZ, joint_pos=joints, eef_pos=eef_positions, eef_rot=eef_orientations)
        break
if __name__ == "__main__":
    main()



