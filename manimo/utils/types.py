"""Custom types for manimo"""
from enum import Enum
from typing import Dict, TypeVar

import numpy as np
import numpy.typing as npt


class ActionSpace(Enum):
    Joint = "Joint"
    Cartesian = "Cartesian"
    JointOnly = "JointOnly"


class IKMode(Enum):
    DMControl = "DMControl"
    Polymetis = "Polymetis"


ObsType = TypeVar("ObsType", np.ndarray, npt.ArrayLike)
ObsDict = Dict[str, ObsType]
