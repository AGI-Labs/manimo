import hydra
from omegaconf import OmegaConf
from manimo.environments.single_arm_env import SingleArmEnv
import time
import torch
import numpy as np

# create a single arm environment

hydra.initialize(
        config_path="../conf", job_name="collect_demos_test"
    )
actuators_cfg = hydra.compose(config_name="actuators")

sensors_cfg = hydra.compose(config_name="sensors")
# sensors_cfg = []

env = SingleArmEnv(sensors_cfg, actuators_cfg)
obs = env.get_obs()
print(f"obs keys: {obs.keys()}")
eef_pos = obs['eef_pos']
eef_quat = obs['eef_rot']
print(f"current eef_pos: {eef_pos}, eef_quat={eef_quat}")
eef_position = eef_pos + np.array([0, 0.01, 0])
print(f"new eef_pos: {eef_position}")
# eef_quat = eef_quat + np.array([0, 0, 0, 1])
action = torch.Tensor(np.concatenate((eef_position, eef_quat)))
print(f"stepping through action")
env.step([action])
time.sleep(5)
# env.step([-action])
obs = env.get_obs()
env.reset()