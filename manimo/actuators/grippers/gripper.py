"""An abstract interface for a gripper which allows users to specify ways to operate the gripper
"""
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from manimo.utils.types import ActionSpace

class Gripper(ABC):
    """Abstract class to move the robot arm
    """
    @abstractmethod
    def __init__(self, gripper_cfg: DictConfig):
        #setup arm with required config
        pass
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass