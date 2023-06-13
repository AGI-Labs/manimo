"""An abstract interface for a sensor 
that allows to users to read values from sensor
"""
from abc import ABC, abstractmethod

from manimo.utils.types import ObsDict


class Sensor(ABC):
    """Abstract class to read values from a sensor"""

    @abstractmethod
    def __init__(self, config):
        """
        Setup sensor with the required config and optional config
        required config:
            sampling rate: number of samples to be collected per second
            buffer_size: number of seconds of data to be retained
        """
        # setup sensor with required config
        pass

    @abstractmethod
    def start(self):
        """
        start polling sensor data
        """

    @abstractmethod
    def close(self):
        """
        stop polling sensor data and close processes
        """

    @abstractmethod
    def reset(self):
        """
        reset sensor
        """
        pass

    # @abstractmethod
    # def set_sampling_rate(hz):
    #     pass

    @abstractmethod
    def get_obs(self) -> ObsDict:
        pass
