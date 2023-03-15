from collections import OrderedDict
import numpy as np
from omegaconf import DictConfig
from reskin_sensor import ReSkinProcess
import time

from manimo.sensors.sensor import Sensor
from manimo.utils.types import ObsDict

class ReskinSensor(Sensor):
    """Abstract class to read values from a sensor
    """
    def __init__(self, config: DictConfig):
        """
        Setup sensor with the required config and optional config
        required config:
            sampling rate: number of samples to be collected per second
            buffer_size: number of seconds of data to be retained
                (The plan is to setup a ring buffer to retain a running collection of recent observed samples)
        """
        super().__init__(config)
        self.config = config
        self.sensor_proc = ReSkinProcess(**config.connection_params)
        self.sensor_proc.start()
        self.connected = True
        self.baseline = None

        try:
            self.baseline_samples = config["num_baseline_samples"]
        except KeyError:
            self.baseline_samples = 100

        # TODO: Fix reskin_sensor to block when starting process
        time.sleep(1.0)
        pass

    def start(self):
        """
        start polling sensor data
        """
        pass
    
    def close(self):
        """
        stop polling sensor data
        """
        if self.connected:
            self.sensor_proc.join()
            self.connected = False

    def _update_baseline(self):
        raw_data = self.sensor_proc.get_data(self.baseline_samples)
        print("baseline_len: ", len(raw_data))
        _, _, reskin_data, _ = zip(*raw_data)
        self.baseline = np.array(reskin_data)
        
    def reset(self):
        # Record baseline and store in default observation
        obs = self.get_obs()

        self._update_baseline()
        info = {"ee_sensing_baseline": self.baseline}

        return obs, info

    def get_obs(self):
        rs_data = self.sensor_proc.get_data(1)
        obs = OrderedDict()
        obs["ee_sensing"] = np.array(rs_data[0].data)
        return obs
