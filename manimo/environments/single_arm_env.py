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
        Initialize the environment with all the actuators and sensors based on the config

        Args:
            sensors_cfg (DictConfig): The config for the sensors
            actuators_cfg (DictConfig): The config for the actuators
            hz (float, optional): The rate at which the environment should run. Defaults to 30.
        """
        self.actuators = [hydra.utils.instantiate(actuators_cfg[actuators][actuator]) for actuators in actuators_cfg for actuator in actuators_cfg[actuators]]
        # self.sensors = [hydra.utils.instantiate(sensors_cfg[sensors][sensor]) for sensors in sensors_cfg for sensor in sensors_cfg[sensors]]
        self.rate = Rate(hz)
    
    def get_obs(self) -> ObsDict:
        """
        Get the observations from all the sensors and actuators

        Returns:
            ObsDict: The observations
        """
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
        """
        Step the environment forward
        Args:
            actions (Optional[np.ndarray], optional): The actions to take. Defaults to None.
        Returns:
            Tuple[ObsDict, float, bool, Dict]: The observations, the reward, whether the episode is done, and any info
        """
        if actions is not None:
            for i, action in enumerate(actions):
                self.actuators[i].step(action)
        self.rate.sleep()
        return self.get_obs(), 0, False, None

    def reset(self):
        """
        Reset the environment
        """
        for actuator in self.actuators:
            actuator.reset()
        # for sensor in self.sensors:
        #     sensor.stop()
        return self.get_obs()
