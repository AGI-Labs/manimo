from typing import Dict, Optional, Tuple

import hydra
import numpy as np
from gym import Env
from omegaconf import DictConfig

from manimo.utils.helpers import Rate
from manimo.utils.types import ObsDict


class SingleArmEnv(Env):
    """
    A single arm manipulation environment with an actuator and various sensors
    """

    def __init__(
        self,
        sensors_cfg: DictConfig,
        actuators_cfg: DictConfig,
        env_cfg: DictConfig,
    ):
        """
        Initialize the environment with all the actuators and
        the sensors based on the config

        Args:
            sensors_cfg (DictConfig): The config for the sensors
            actuators_cfg (DictConfig): The config for the actuators
            hz (float, optional): The rate at which the environment should run.
            Defaults to 30.
        """
        # TODO: Add support for accessing the actuators and sensors by name
        self.actuators = [
            hydra.utils.instantiate(actuators_cfg[actuator_type][actuator])
            for actuator_type in actuators_cfg
            for actuator in actuators_cfg[actuator_type]
        ]
        self.sensors = []
        if sensors_cfg:
            self.sensors = [
                hydra.utils.instantiate(sensors_cfg[sensor_type][sensor])
                for sensor_type in sensors_cfg
                for sensor in sensors_cfg[sensor_type]
            ]

        self.rate = Rate(env_cfg.hz)

    def get_obs(self) -> ObsDict:
        """
        Get the observations from all the sensors and actuators

        Returns:
            ObsDict: The observations
        """
        obs = {}
        # Aggregate observations from the sensors
        for sensor in self.sensors:
            obs.update(sensor.get_obs())

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
            actions (Optional[np.ndarray], optional):
            The actions to take. Defaults to None.
        Returns:
            Tuple[ObsDict, float, bool, Dict]:
            The observations, the reward, whether the episode is done,
            and any info
        """
        obs = {}
        if actions is not None:
            for i, action in enumerate(actions):
                # self.actuators[i].step(action)
                obs.update(self.actuators[i].step(action))
        self.rate.sleep()
        obs.update(self.get_obs())
        return obs, 0, False, None

    def reset(self):
        """
        Reset the environment
        """
        obs = dict()
        info = dict()
        for actuator in self.actuators:
            act_obs, act_info = actuator.reset()
            obs.update(act_obs)
            info.update(act_info)
        for sensor in self.sensors:
            sens_obs, sens_info = sensor.reset()
            obs.update(sens_obs)
            info.update(sens_info)

        return obs, info

    def close(self):
        """
        Resets the actuators and closes the sensors
        """
        for actuator in self.actuators:
            act_obs, act_info = actuator.reset()
        for sensor in self.sensors:
            sensor.close()

    def set_home(self, new_home=None):
        """
        Go home for all the actuators

        Args:
            new_home (Optional[np.ndarray], optional): The new home position.
            Defaults to None.
        """
        for actuator in self.actuators:
            actuator_set_home = getattr(actuator, "set_home", None)
            if callable(actuator_set_home):
                actuator_set_home(new_home)
