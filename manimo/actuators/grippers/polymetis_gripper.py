import numpy as np
from gym import spaces
from manimo.actuators.grippers.gripper import Gripper
from omegaconf import DictConfig
from polymetis import GripperInterface


class PolymetisGripper(Gripper):
    """
    A class to control the Franka Emika Panda gripper
    """

    def __init__(self, gripper_cfg: DictConfig):
        """
        Initialize the gripper
        Args:
            gripper_cfg (DictConfig): The config for the gripper
        """
        self.config = gripper_cfg
        self._gripper_interface = GripperInterface(
            ip_address=gripper_cfg.server.ip_address,
            # port=gripper_cfg.server.port,
        )
        print(f"connection to gripper established!")
        self.action_space = spaces.Box(
            0.0,
            self._gripper_interface.metadata.max_width,
            (1,),
            dtype=np.float32,
        )

    def _open_gripper(self):
        max_width = self._gripper_interface.metadata.max_width
        self._gripper_interface.goto(
            width=max_width,
            speed=self.config.speed,
            force=self.config.force,
        )

    def _close_gripper(self):
        self._gripper_interface.grasp(
            speed=self.config.speed, force=self.config.force
        )

    def step(self, action):
        obs = {}
        if action is not None:
            gripper_close_width = max(
                self.action_space.high[0] * self.config.close_width_pct,
                self.action_space.low[0],
            )
            action = np.clip(
                action, gripper_close_width, self.action_space.high[0]
            )
            self._gripper_interface.goto(
                width=action,
                speed=self.config.speed,
                force=self.config.force,
            )
        obs["eef_gripper_action"] = action
        return obs

    def reset(self):
        """
        Reset the gripper to the initial state
        """
        self._open_gripper()
        print("gripper reset")
        return self.get_obs(), {}

    def get_obs(self):
        """
        Get the observations from the gripper
        Returns:
            ObsDict: The observations
        """
        obs = {}
        obs["eef_gripper_width"] = self._gripper_interface.get_state().width
        return obs
