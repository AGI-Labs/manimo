import abc
import typing

import numpy as np


class TeleopDeviceReader:
    """Allows for teleoperation using either the keyboard
    or an Oculus controller
    """

    @abc.abstractmethod
    def get_state(self) -> typing.Tuple[bool, np.array, bool]:
        raise NotImplementedError
