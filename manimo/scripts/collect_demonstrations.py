import hydra
from omegaconf import OmegaConf
from manimo.environments.single_arm_env import SingleArmEnv
import torch

# create a single arm environment

hydra.initialize(
        config_path="../conf", job_name="collect_demos_test"
    )
actuators_cfg = hydra.compose(config_name="config")

sensors_config = ""

env = SingleArmEnv(sensors_config=sensors_config, actuators_config=actuators_cfg)
action = torch.Tensor([0, 0, -0.1])
env.step([action])
