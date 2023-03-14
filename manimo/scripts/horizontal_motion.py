import faulthandler
import hydra
from omegaconf import OmegaConf
from manimo.environments.single_arm_env import SingleArmEnv
import time
import numpy as np
import torch

# create a single arm environment

hydra.initialize(
        config_path="../conf", job_name="collect_demos_test"
    )
actuators_cfg = hydra.compose(config_name="actuators")
sensors_cfg = hydra.compose(config_name="sensors")
# sensors_cfg = []

faulthandler.enable()
env = SingleArmEnv(sensors_cfg, actuators_cfg)
env.reset()

start_time = time.time()
total_time = 60
delta = 1e-2

while time.time() - start_time < total_time:
    print(time.time())
    try:
        if time.time() - start_time < 5:
            delta_action = np.array([0, delta, 0, 0, 0, 0, 1])
        elif time.time() - start_time < 10:
            delta_action = np.array([0, -delta, 0, 0, 0, 0, 1])
        else:
            delta_action = np.array([0, -delta, 0, 0, 0, 0, 1])
    except:
        pass
    delta_action = delta_action.reshape(1, -1)


    env.step(delta_action)
