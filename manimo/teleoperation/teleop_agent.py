import hydra
import numpy as np
from manimo.utils.types import ObsDict
import torch
from scipy.spatial.transform import Rotation as R
from typing import Optional

def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler('xyz', degrees=degrees)
    return euler

def euler_to_quat(euler, degrees=False):
    return R.from_euler('xyz', euler, degrees=degrees).as_quat()

def quat_diff(target, source):
    result = R.from_quat(target) * R.from_quat(source).inv()
    return result.as_quat()

def quat_add(target, source):
    result = R.from_quat(target) * R.from_quat(source)
    return result.as_quat()

class Agent:
    """
    An abstract agent class
    """
    def __init__(self):
        pass

    def get_action(self, obs: ObsDict) -> np.ndarray:
        """
        Get the action from the agent

        Args:
            obs (ObsDict): The observations

        Returns:
            np.ndarray: The action
        """
        raise NotImplementedError


class TeleopAgent(Agent):
    """
    A teleoperation agent
    """
    def __init__(self):
        """
        Args:
            robot (Robot): The robot
        """
        hydra.initialize(config_path="./conf", job_name="teleop_agent")
        self.teleop_cfg = hydra.compose(config_name="teleop_config")
        self.teleop = hydra.utils.instantiate(self.teleop_cfg.device)['device']
        
        # Initialize variables
        self.robot_origin = {'pos': None, 'quat': None}
        self.vr_origin = {'pos': None, 'quat': None}
        self.init_ref = True

        self.pos_action_gain = 0.5
        self.rot_action_gain = 0.2
        self.gripper_action_gain = 1

    def get_action(self, obs: ObsDict) -> Optional[np.ndarray]:
        """
        Get the action from the agent

        Args:
            obs (ObsDict): The observations

        Returns:
            np.ndarray: The action
        """
        # Obtain info from teleop device
        control_en, grasp_en, vr_pose_curr = self.teleop.get_state()

        print(f"Control: {control_en}, Grasp: {grasp_en}, Pose: {vr_pose_curr}")

        vr_pos, vr_quat = vr_pose_curr

        robot_pos = obs['eef_pos']
        robot_quat = obs['eef_rot']
        # robot_gripper_width = obs['eef_gripper_width']

        try:
            # Update arm
            if control_en:
                # Update reference pose
                if self.init_ref:
                    self.robot_origin = {'pos': robot_pos, 'quat': robot_quat}
                    self.vr_origin = {'pos': vr_pos, 'quat': vr_quat}
                    self.init_ref = False

                # Calculate Positional Action
                target_pos_offset = vr_pos - self.vr_origin['pos']
                pos_action = target_pos_offset

                # Calculate Euler Orientation Action
                target_quat_offset = quat_diff(vr_quat, self.vr_origin['quat'])
                quat_action = target_quat_offset
                euler_action = quat_to_euler(quat_action)

                # Scale Appropriately
                pos_action *= self.pos_action_gain
                euler_action *= self.rot_action_gain

                # Add robot origin
                quat_action = euler_to_quat(euler_action)
                quat_action = quat_add(self.robot_origin['quat'], quat_action)
                pos_action += self.robot_origin['pos']
                        
                
                action = np.append(pos_action, quat_action), grasp_en
                
                return action

            else:
                self.init_ref = True
                return None

        except KeyboardInterrupt:
            print("Session ended by user.")
            return None