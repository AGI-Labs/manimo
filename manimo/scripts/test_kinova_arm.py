import faulthandler

import hydra
from manimo.environments.single_arm_env import SingleArmEnv
import time
# create a single arm environment

hydra.initialize(config_path="../conf", job_name="collect_demos_test")
actuators_cfg = hydra.compose(config_name="actuators")
sensors_cfg = hydra.compose(config_name="sensors")
env_cfg = hydra.compose(config_name="env")
# sensors_cfg = []
# actuators_cfg = []
faulthandler.enable()
env = SingleArmEnv(sensors_cfg, actuators_cfg, env_cfg)
env.reset()
obs = env.get_obs()
print(obs)
time.sleep(2)
# close
for _ in range(5):
    env.step([[0.1, 0, 0, 0, 0, 0]])

for _ in range(5):
    env.step([[0, 0.1, 0, 0, 0, 0]])

# open
for _ in range(5):
    env.step([[0, 0, 0.1, 0, 0, 0]])


env.reset()

for key in obs:
    print(f"obs key: {key}, values: {obs[key]}")
