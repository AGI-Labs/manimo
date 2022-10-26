from gym import Env
import hydra
import numpy as np
from omegaconf import DictConfig
from typing import Dict, Optional, Tuple
from manimo.utils.types import ObsDict


class SingleArmEnv(Env):
    """
    A single arm manipulation environment with an actuator and various sensors
    """
    def __init__(self, sensors_cfg: DictConfig, actuators_config: DictConfig):
        """
        Initialize all the actuators and sensors based on the config
        """
        self.actuators = [hydra.utils.instantiate(actuators_config[actuator]) for actuator in actuators_config.keys()]
        self.sensors = [hydra.utils.instantiate(sensors_cfg[sensor]) for sensor in sensors_cfg.keys()]

    def get_obs(self):
        obs = {}
        for sensor in self.sensors:
            obs.update(sensor.get_obs())
        return obs
        
    def step(
        self, actions: Optional[np.ndarray] = None
    ) -> Tuple[ObsDict, float, bool, Dict]:
        for i, action in enumerate(actions):
            self.actuators[i].step(action)

    def reset(self):
        for sensor in self.sensors:
            sensor.stop()
