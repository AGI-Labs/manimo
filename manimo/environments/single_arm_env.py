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
        arm = hydra.utils.instantiate(actuators_config.arm)
        self.actuators.append(arm)

    def step(
        self, actions: Optional[np.ndarray] = None
    ) -> Tuple[ObsDict, float, bool, Dict]:
        for i, action in enumerate(actions):
            self.actuators[i].step(action)

