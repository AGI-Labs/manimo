import hydra
import numpy as np
from manimo.utils.types import ObsDict
import sophus as sp
import torch
from torchcontrol.transform import Rotation as R
from typing import Optional

def pose_elementwise_diff(pose1, pose2):
    return sp.SE3(
        (pose2.so3() * pose1.so3().inverse()).matrix().T,
        pose2.translation() - pose1.translation(),
    )

def pose_elementwise_apply(delta_pose, pose):
    return sp.SE3(
        (pose.so3() * delta_pose.so3()).matrix(),
        delta_pose.translation() + pose.translation(),
    )

def getSE3Pose(pos, quat):
        rotvec = R.from_quat(quat).as_rotvec()
        return sp.SE3(sp.SO3.exp(rotvec).matrix(), pos)

def getPosQuat(pose_des):
    # Compute desired pos & quat
    ee_pos_desired = torch.Tensor(pose_des.translation())
    ee_quat_desired = R.from_matrix(
        torch.Tensor(pose_des.rotationMatrix())
    ).as_quat()
    return ee_pos_desired, ee_quat_desired

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
        self.vr_pose_ref = sp.SE3()
        self.arm_pose_ref = sp.SE3()
        self.init_ref = True


    def get_action(self, obs: ObsDict) -> Optional[np.ndarray]:
        """
        Get the action from the agent

        Args:
            obs (ObsDict): The observations

        Returns:
            np.ndarray: The action
        """
        # Obtain info from teleop device
        is_active, vr_pose_curr, grasp_state = self.teleop.get_state()

        try:
            # Update arm
            print(f"is_active: {is_active}")
            if is_active:
                # Update reference pose
                if self.init_ref:
                    self.vr_pose_ref = vr_pose_curr
                    self.arm_pose_ref = getSE3Pose(obs['eef_pos'], torch.Tensor(obs['eef_rot']))
                    self.init_ref = False

                # Determine pose
                vr_pose_diff = pose_elementwise_diff(self.vr_pose_ref, vr_pose_curr)
                vr_pose_diff = sp.SE3.exp(self.teleop_cfg.sensitivity * vr_pose_diff.log())
                arm_pose_desired = pose_elementwise_apply(vr_pose_diff, self.arm_pose_ref)

                ee_pos_desired, ee_quat_desired = getPosQuat(arm_pose_desired)
                return np.append(ee_pos_desired.numpy(), ee_quat_desired.numpy()), grasp_state

            else:
                self.init_ref = True
                return None

        except KeyboardInterrupt:
            print("Session ended by user.")
            return None