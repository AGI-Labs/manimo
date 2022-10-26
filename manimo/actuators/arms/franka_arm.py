import grpc
import hydra
from manimo.actuators.arms.arm import Arm
from manimo.actuators.arms.moma_arm import MujocoArmModel
from manimo.actuators.controllers import CartesianPDPolicy, JointPDPolicy
from manimo.utils.types import ActionSpace, IKMode
import numpy as np
from omegaconf import DictConfig
from polymetis import RobotInterface
import time
import torch

class FrankaArm(Arm):
    def __init__(self, arm_cfg: DictConfig):
        self.config = arm_cfg
        self.action_space = ActionSpace(arm_cfg.action_space)
        self.delta = arm_cfg.delta
        self.hz = arm_cfg.hz
        self.ik_mode = IKMode(arm_cfg.ik_mode)
        self.JOINT_LIMIT_MIN = arm_cfg.joint_limit_min
        self.JOINT_LIMIT_MAX = arm_cfg.joint_limit_max
        self.robot = RobotInterface(ip_address=self.config.robot_ip, enforce_version=False)

        self._setup_mujoco_ik()
        self.connect()
    
    def connect(self, policy=None, wait=2):
        if policy is None:
            policy = self._default_policy()
        self.policy = policy
        self.robot.send_torch_policy(policy, blocking=False)
        time.sleep(wait)

    def _default_policy(self, kq_ratio=1.5, kqd_ratio=1.5):
        q_initial = self.robot.get_joint_positions()
        kq = kq_ratio * torch.Tensor(self.robot.metadata.default_Kq)
        kqd = kqd_ratio * torch.Tensor(self.robot.metadata.default_Kqd)
        kx=torch.Tensor(self.robot.metadata.default_Kx)
        kxd=torch.Tensor(self.robot.metadata.default_Kxd)

        if self.action_space == ActionSpace.Joint:
            return JointPDPolicy(
                    desired_joint_pos=torch.tensor(q_initial),
                    kq=kq, kqd=kqd,
            )
        elif self.action_space == ActionSpace.Cartesian:
            if self.ik_mode == IKMode.Polymetis:
                return CartesianPDPolicy(
                torch.tensor(q_initial), True, kq, kqd, kx, kxd)
            elif self.ik_mode == IKMode.DMControl:
                return JointPDPolicy(
                    desired_joint_pos=torch.tensor(q_initial),
                    kq=kq, kqd=kqd,
            )

    def _setup_mujoco_ik(self):
        self.mujoco_model = MujocoArmModel(self.config)

    def _apply_joint_commands(self, q_desired):
        q_des_tensor = np.array(q_desired)
        q_des_tensor = torch.tensor(np.clip(
            q_des_tensor, self.JOINT_LIMIT_MIN, self.JOINT_LIMIT_MAX))
        self.robot.update_current_policy({"q_desired": q_des_tensor.float()})

    def _get_desired_pos_quat(self, eef_pose):
        if self.delta:
            ee_pos_cur, ee_quat_cur = self.robot.get_ee_pose()
            ee_pos_desired = ee_pos_cur + torch.Tensor(eef_pose[:3])
            ee_quat_desired = ee_quat_cur + torch.Tensor(eef_pose[3:])  
        else:
            ee_pos_desired = torch.Tensor(eef_pose[:3])
            ee_quat_desired = torch.Tensor(eef_pose[3:])

        return ee_pos_desired, ee_quat_desired

    def _apply_eef_commands(self, eef_pose, wait_time=3):

        ee_pos_desired, ee_quat_desired = self._get_desired_pos_quat()

        joint_pos_cur = self.robot.get_joint_positions(eef_pose)
        joint_pos_desired, success = self.robot.solve_inverse_kinematics(
            ee_pos_desired, ee_quat_desired, joint_pos_cur
        )
        joint_pos_desired = joint_pos_desired.numpy()
        
        update_success = True
        try:
            joint_pos_desired = torch.tensor(np.clip(
            joint_pos_desired, self.JOINT_LIMIT_MIN, self.JOINT_LIMIT_MAX))
            self.robot.update_current_policy({"joint_pos_desired": joint_pos_desired.float()})
        except grpc.RpcError:
            self.robot.send_torch_policy(self.policy, blocking=False)
            update_success = False
            time.sleep(wait_time)
        
        return update_success

    
    def step(self, action):
        if self.action_space == ActionSpace.Cartesian:
            if self.ik_mode == IKMode.Polymetis:
                self._apply_eef_commands(action)

            elif self.ik_mode == IKMode.DMControl:
                ee_pos_current, ee_quat_current = self.robot.get_ee_pose()
                cur_joint_positions = self.robot.get_joint_positions().numpy()
                ee_pos_desired, ee_quat_desired = self._get_desired_pos_quat(action)
                desired_action, _ = self.mujoco_model.local_inverse_kinematics(ee_pos_desired, ee_quat_desired, ee_pos_current, ee_quat_current, cur_joint_positions)
                self._apply_joint_commands(desired_action)