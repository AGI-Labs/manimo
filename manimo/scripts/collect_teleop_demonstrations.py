import argparse
import asyncio
import glob
import hydra
from manimo.environments.single_arm_env import SingleArmEnv
from manimo.teleoperation.teleop_agent import TeleopAgent
from manimo.utils.helpers import HOMES
from multiprocessing import Process, Queue
import numpy as np
import os
from PIL import Image
import time

# create a single arm environment
def _create_data_folder(dir, input, task, index):
    data_folder = f"{dir}/{input}_{task}_{index}"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    return data_folder

def dump_images(imgs_queue: Queue, data_folder: str):
    step = 0
    while True:
        try:
            imgs = imgs_queue.get()
            imgs = [Image.fromarray(img) for img in imgs]
            for i, image in enumerate(imgs):
                print(f"saving image {step}_cam{i}.jpg")
                image.save(f"{data_folder}/{step}_cam{i}.jpg")
            step += 1
        except:
            pass


def _get_filename(dir, input, task):
    index = 0
    for name in glob.glob("{}/{}_{}_*.npz".format(dir, input, task)):
        n = int(name[:-4].split("_")[-1])
        if n >= index:
            index = n + 1
    return "{}/{}_{}_{}.npz".format(dir, input, task, index), index

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--task", type=str, default="insertion")
    parser.add_argument("--time", type=float, default=100)
    parser.add_argument("--path", type=str, default="data")
    parser.add_argument("--gripper_en", type=bool, default= False)
    imgs_queue = Queue(maxsize=1)
    # parser.add_argument("--")

    args = parser.parse_args()
    name = args.name
    task = args.task
    TIME = args.time
    HZ = 5
    home = HOMES[task]

    hydra.initialize(
            config_path="../conf", job_name="collect_demos_test"
        )

    actuators_cfg = hydra.compose(config_name="actuators")
    sensors_cfg = hydra.compose(config_name="sensors")

    env = SingleArmEnv(sensors_cfg, actuators_cfg)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    agent = TeleopAgent()

    obs, _ = env.reset()

    while True:
        filename, index = _get_filename(args.path, name, task)
        data_folder = _create_data_folder(args.path, name, task, index)
        # make a directory if not exists with args.path + name
        logger_proc = Process(target=dump_images, args=(imgs_queue, data_folder), daemon=True)
        logger_proc.start()        


        user_in = "r"
        # while user_in == "r":
        obs, _ = env.reset()
        # user_in = input("Ready. Recording {}".format(filename))
        

        joints = []
        eef_positions = []
        eef_orientations = []
        step = 0
        for _ in range(int(TIME * HZ) - 1):
            start = time.time()

            action = agent.get_action(obs)
            if action is not None:
                arm_action, gripper_action = action
                if args.gripper_en:
                    action = [arm_action, gripper_action]
                else:
                    action = [arm_action]
            obs = env.step(action)[0]
            
            # store observations
            time_before_dump = time.time() - start
            if action is None:
                # store images
                imgs = [obs[key][0] for key in obs if 'cam' in key]
                try:
                    imgs_queue.put(imgs, block=True)
                except:
                    try:
                        imgs_queue.get(block=False)
                    except:
                        pass
                    
                joints.append(obs["q_pos"])
                eef_positions.append(obs["eef_pos"])
                eef_orientations.append(obs["eef_rot"])
                step += 1
            
            end = time.time() - start

            # print(f"took {end} seconds to collect data, {time_before_dump} seconds to obtain observations")
        
        # store the last observation
        env.reset()

        logger_proc.terminate()

        if not os.path.exists("./data"):
            os.mkdir("data")

        print(f"created new directory!")
        np.savez(filename, home=home, hz=HZ, joint_pos=joints, eef_pos=eef_positions, eef_rot=eef_orientations)
        break
if __name__ == "__main__":
    main()