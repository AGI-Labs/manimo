from abc import ABC, abstractmethod

"""
Contains basic callbacks for manimo.
"""

class BaseCallback(ABC):
    """
    Base class for all callbacks.
    """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger


    @abstractmethod
    def on_begin_traj(self, traj_idx):
        """
        Called at the beginning of a trajectory.
        """
        pass

    @abstractmethod
    def on_end_traj(self, traj_idx):
        """
        Called at the end of a trajectory.
        """
        pass

    @abstractmethod
    def on_step(self, traj_idx, step_idx) -> bool:
        """
        Called at the end of each step.
        """
        pass