"""An abstract interface for a sensor that allows to users to read values from sensor
"""
from abc import ABC, abstractmethod
from manimo.utils.types import ObsDict

class Sensor(ABC):
    """Abstract class to read values from a sensor
    """
    @abstractmethod
    def __init__(self, config):
        """
        Setup sensor with the required config and optional config
        required config:
            sampling rate: number of samples to be collected per second
            buffer_size: number of seconds of data to be retained
                (The plan is to setup a ring buffer to retain a running collection of recent observed samples)
        """
        #setup sensor with required config
        pass

    @abstractmethod
    def start(self):
        """
        start polling sensor data
        """
    
    @abstractmethod
    def stop(self):
        """
        stop polling sensor data
        """

    # @abstractmethod
    # def set_sampling_rate(hz):
    #     pass

    @abstractmethod
    def get_obs(self) -> ObsDict:
        pass
