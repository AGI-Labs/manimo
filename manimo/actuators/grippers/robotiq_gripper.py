from gym import spaces
from polymetis.robot_client.robotiq_gripper import RobotiqGripperClient
from manimo.actuators.grippers.gripper import Gripper
import numpy as np
from omegaconf import DictConfig

class Robotiq2fGripper(Gripper):
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
        self._gripper_interface = RobotiqGripperClient(gripper_cfg.server.ip_address, gripper_cfg.server.port, gripper_cfg.server.comport)
        print(f"connection to gripper established!")
        # Query for command
        self.cmd = self._gripper_interface.connection.ControlUpdate(self._gripper_interface.get_gripper_state())
        self.cmd.speed = self.config.speed
        self.cmd.force = self.config.force
        self.action_space = spaces.Box(
            0.0, self._gripper_interface.metadata.max_width, (1,), dtype=np.float32
        )
        self._gripper_interface.run()

    def _open_gripper(self):
        self._gripper_interface.apply_gripper_command(
            self.cmd
        )

    def _close_gripper(self):
        self.cmd.width = 0
        
        self._gripper_interface.grasp(
            self.cmd
        )

    def step(self, action):
        if action is not None:
            if action:
                self._close_gripper()
            else:
                self._open_gripper()

    def reset(self):
        """
        Reset the gripper to the initial state
        """
        self._open_gripper()
        return self.get_obs(), {}


    def get_obs(self):
        """
        Get the observations from the gripper
        Returns:
            ObsDict: The observations
        """
        obs = {}
        obs["eef_gripper_width"] = self._gripper_interface.get_gripper_state().width
        return obs
    

    


