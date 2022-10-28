import hydra
from omegaconf import OmegaConf
from manimo.environments.single_arm_env import SingleArmEnv
import time
import torch

# create a single arm environment

hydra.initialize(
        config_path="../conf", job_name="collect_demos_test"
    )
actuators_cfg = hydra.compose(config_name="actuators")

sensors_cfg = hydra.compose(config_name="sensors")
# sensors_cfg = []

env = SingleArmEnv(sensors_cfg, actuators_cfg)
eef_position = [0, 0, 0.05]
eef_orientation = [0., 0., 0., 0.]
action = torch.Tensor(eef_position + eef_orientation)
env.step([action])
time.sleep(5)
env.step([-action])
obs = env.get_obs()
env.reset()

for key in obs:
    print(f"obs key: {key}, data: {obs[key]}")