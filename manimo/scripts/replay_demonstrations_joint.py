import argparse

import h5py
import hydra

from manimo.environments.single_arm_env import SingleArmEnv

parser = argparse.ArgumentParser()
parser.add_argument("--replay_file", type=str, required=True)
args = parser.parse_args()
replay_file = args.replay_file

# read the test.h5 file
f = h5py.File(replay_file, "r")

# get position action
ee_pos_desired = f["eef_pos"][:]
ee_quat_desired = f["eef_rot"][:]
ee_joints = f["q_pos"][:]
ee_joint_action = f["joint_action"][:]
gripper_width = f["eef_gripper_width"][:]

# import pdb; pdb.set_trace()

# get start position
demo_start_pos = ee_joint_action[0]
print(f"num_obs: {ee_pos_desired.shape[0]}")
print(f"start_pos: {demo_start_pos}")

# initialize agent
hydra.initialize(config_path="../conf", job_name="collect_demos_test")

actuators_cfg = hydra.compose(config_name="actuators_playback")
sensors_cfg = hydra.compose(config_name="sensors")
env_cfg = hydra.compose(config_name="env")

env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg=env_cfg)
hydra.core.global_hydra.GlobalHydra.instance().clear()

print(f"Setting home to demo start pos: {demo_start_pos}")
env.set_home(demo_start_pos)
env.reset()

assert ee_pos_desired.shape[0] == ee_quat_desired.shape[0]
num_actions = ee_pos_desired.shape[0]

print("starting replay")

for i in range(num_actions):
    action = [ee_joint_action[i], gripper_width[i]]
    env.step(action)

env.reset()
