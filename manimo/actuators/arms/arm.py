"""An abstract interface for arm which allows users to specify ways to move the arm
   and actually move the arm
"""
from abc import ABC, abstractmethod

class Arm(ABC):
    """Abstract class to move the robot arm
    """
    @abstractmethod
    def __init__(self, config):
        #setup arm with required config
        pass

    @abstractmethod
    def step(action):
        pass
