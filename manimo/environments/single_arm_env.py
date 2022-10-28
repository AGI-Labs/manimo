from gym import Env
import hydra
import numpy as np
from omegaconf import DictConfig
from typing import Dict, Optional, Tuple
from manimo.utils.helpers import Rate
from manimo.utils.types import ObsDict


class SingleArmEnv(Env):
    """
    A single arm manipulation environment with an actuator and various sensors
    """
    def __init__(self, sensors_cfg: DictConfig, actuators_cfg: DictConfig, hz=30.):
        """
        Initialize all the actuators and sensors based on the config
        """
        self.actuators = [hydra.utils.instantiate(actuators_cfg[actuator]) for actuator in actuators_cfg.keys()]
        self.sensors = [hydra.utils.instantiate(sensors_cfg[sensor]) for sensor in sensors_cfg.keys()]
        self.rate = Rate(hz)
    
    def get_obs(self):
        obs = {}
        # Aggregate observations from the sensors
        # for sensor in self.sensors:
        #     obs.update(sensor.get_obs())

        # Some of the actuators can also have observations
        for actuator in self.actuators:
            obs.update(actuator.get_obs())
        return obs
        
    def step(
        self, actions: Optional[np.ndarray] = None
    ) -> Tuple[ObsDict, float, bool, Dict]:
        if actions is not None:
            for i, action in enumerate(actions):
                self.actuators[i].step(action)
        self.rate.sleep()
        return self.get_obs(), 0, False, None

    def reset(self):
        for actuator in self.actuators:
            actuator.reset()
        # for sensor in self.sensors:
        #     sensor.stop()
        return self.get_obs()
