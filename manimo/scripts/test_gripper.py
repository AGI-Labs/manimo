import faulthandler

import hydra
from manimo.environments.single_arm_env import SingleArmEnv

# create a single arm environment

hydra.initialize(config_path="../conf", job_name="collect_demos_test")
actuators_cfg = hydra.compose(config_name="actuators_gripper")
sensors_cfg = hydra.compose(config_name="sensors")
sensors_cfg = []

faulthandler.enable()
env = SingleArmEnv(sensors_cfg, actuators_cfg)
obs = env.get_obs()

# close
for _ in range(20):
    env.step([1])

# open
for _ in range(20):
    env.step([0])


env.reset()

for key in obs:
    print(f"obs key: {key}, values: {obs[key]}")
