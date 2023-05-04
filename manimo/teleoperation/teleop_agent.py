import hydra
import numpy as np
from manimo.utils.types import ObsDict
import torch
from scipy.spatial.transform import Rotation as R
from typing import Optional

action_scaler = 10

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
        self.use_gripper = self.teleop_cfg.use_gripper
        self.disable_rot = self.teleop_cfg.disable_rot
        self.position_mask = self.teleop_cfg.position_mask

        self.pos_action_gain = 3
        self.rot_action_gain = 6
        self.gripper_action_gain = 1
        self.max_lin_vel = 1
        self.max_rot_vel = 1

    def _limit_velocity(self, lin_vel, rot_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        
        return lin_vel, rot_vel

    def get_action(self, obs: ObsDict) -> Optional[np.ndarray]:
        """
        Get the action from the agent

        Args:
            obs (ObsDict): The observations
            apply_pos_mask (bool, optional): Whether to apply the position mask. Defaults to True.

        Returns:
            np.ndarray: The action
        """
        # Obtain info from teleop device
        control_en, grasp_en, vr_pose_curr, buttons = self.teleop.get_state()
        vr_pos, vr_quat = vr_pose_curr
        robot_pos = obs['eef_pos']
        robot_quat = obs['eef_rot']

        # option to use gripper
        if self.use_gripper:
            # TODO: implement gripper actions
            robot_gripper_width = obs['eef_gripper_width']

        try:
            # Update arm
            if control_en:
                # Update reference pose
                if self.init_ref:
                    self.robot_origin = {'pos': robot_pos, 'quat': robot_quat}
                    self.vr_origin = {'pos': vr_pos, 'quat': vr_quat}
                    self.init_ref = False

                # Calculate Positional Action
                robot_pos_offset = robot_pos - self.robot_origin['pos']
                target_pos_offset = vr_pos - self.vr_origin['pos']
                pos_action = target_pos_offset - robot_pos_offset

                # Calculate Euler Orientation Action
                target_quat_offset = quat_diff(vr_quat, self.vr_origin['quat'])
                quat_action = target_quat_offset
                euler_action = quat_to_euler(quat_action)

                # Add robot origin
                quat_action = euler_to_quat(euler_action)

                 # Calculate Euler Action #
                robot_quat_offset = quat_diff(robot_quat, self.robot_origin['quat'])
                target_quat_offset = quat_diff(vr_quat, self.vr_origin['quat'])
                quat_action = quat_diff(target_quat_offset, robot_quat_offset)
                # quat_action = robot_quat_offset
                euler_action = quat_to_euler(quat_action)

                # Scale Appropriately
                pos_action *= self.pos_action_gain
                euler_action *= self.rot_action_gain

                # pos_action += self.robot_origin['pos']    
                #arm_action, gripper_action = np.append(pos_action, quat_action), grasp_en
                pos_vel, euler_vel = self._limit_velocity(pos_action, euler_action)
                # arm_action, gripper_action = np.append(pos_action, euler_action), grasp_en
                arm_action, gripper_action = np.append(pos_vel, euler_vel), grasp_en

                
                return arm_action*action_scaler, gripper_action, buttons

            else:
                self.init_ref = True
                return None, None, buttons

        except KeyboardInterrupt:
            print("Session ended by user.")
            return None