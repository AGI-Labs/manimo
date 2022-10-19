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
    def __init__(self, sensors_config: DictConfig, actuators_config: DictConfig):
        """
        TODO (Mohan): How can we instantiate different sensor classes from a list
        of configs - sensors_config
        """
        self.actuators = []
        # arm = hydra.utils.instantiate(actuators_config.arm)
        # self.actuators.append(arm)

        self.sensors = []
        camera = hydra.utils.instantiate(sensors_config.camera)
        self.sensors.append(camera)

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
