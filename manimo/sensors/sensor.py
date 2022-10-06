"""An abstract interface for a sensor that allows to users to read values from sensor
"""
from abc import ABC, abstractmethod
from utils.types import Observation

class Sensor(ABC):
    """Abstract class to read values from a sensor
    """
    @abstractmethod
    def __init__(self, config):
        #setup sensor with required config
        pass

    @abstractmethod
    def get_obs() -> Observation:
        pass
