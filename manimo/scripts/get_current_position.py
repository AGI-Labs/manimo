import faulthandler

import hydra
from manimo.environments.single_arm_env import SingleArmEnv

# create a single arm environment

hydra.initialize(config_path="../conf", job_name="collect_demos_test")
env_cfg = hydra.compose(config_name="env")
actuators_cfg = hydra.compose(config_name="actuators")
sensors_cfg = hydra.compose(config_name="sensors")

faulthandler.enable()
env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg)
obs = env.get_obs()

# env.reset()

for key in obs:
    print(f"obs key: {key}, values: {obs[key]}")

env.reset()
