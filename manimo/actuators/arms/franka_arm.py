from argparse import Action
import grpc
import hydra
from manimo.actuators.arms.arm import Arm
from manimo.actuators.arms.moma_arm import MujocoArmModel
from manimo.actuators.controllers import CartesianPDPolicy, JointPDPolicy
from manimo.utils.helpers import Rate
from manimo.utils.types import ActionSpace, IKMode
from manimo.teleoperation.teleop_agent import quat_add
import numpy as np
from omegaconf import DictConfig
from polymetis import RobotInterface
import time
import torch
import torchcontrol as toco

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
        self.kq = arm_cfg.kq
        self.kqd = arm_cfg.kqd
        self.home = arm_cfg.home if arm_cfg.home is not None else self.robot.get_joint_positions()
        self._setup_mujoco_ik()
        self.reset()
    
    def connect(self, policy=None, wait=2):
        if policy is None:
            policy = self._default_policy(self.action_space)
        self.policy = policy
        self.robot.send_torch_policy(policy, blocking=False)
        time.sleep(wait)

    def reset(self):
        self._go_home()
        self.connect()

        obs = self.get_obs()
        return obs, {}

    def _go_home(self):
        # Create policy instance
        q_initial = self.robot.get_joint_positions()
        waypoints = toco.planning.generate_joint_space_min_jerk(
            start=q_initial,
            goal=torch.Tensor(self.home),
            time_to_go=4,
            hz=50)
        hz = 50.0
        rate = Rate(hz)
        joint_positions = [waypoint["position"] for waypoint in waypoints]

        q_initial = self.robot.get_joint_positions()
        kq = torch.Tensor(self.robot.metadata.default_Kq)
        kqd = torch.Tensor(self.robot.metadata.default_Kqd)
        policy = JointPDPolicy(
                    desired_joint_pos=q_initial,
                    kq=kq, kqd=kqd,)
        self.robot.send_torch_policy(policy, blocking=False)
        rate.sleep()
        for joint_position in joint_positions:
            self.robot.update_current_policy(
                        {"q_desired": joint_position})
            rate.sleep()

    def _default_policy(self, action_space, kq_ratio=1.5, kqd_ratio=1.5):
        q_initial = self.robot.get_joint_positions()
        kq = kq_ratio * torch.Tensor(self.kq)
        kqd = kqd_ratio * torch.Tensor(self.kqd)
        kx=torch.Tensor(self.robot.metadata.default_Kx)
        kxd=torch.Tensor(self.robot.metadata.default_Kxd)

        if action_space == ActionSpace.Joint:
            return JointPDPolicy(
                    desired_joint_pos=q_initial,
                    kq=kq, kqd=kqd,
            )
        elif action_space == ActionSpace.Cartesian:
            if self.ik_mode == IKMode.Polymetis:
                return CartesianPDPolicy(q_initial, True, kq, kqd, kx, kxd)
            elif self.ik_mode == IKMode.DMControl:
                return JointPDPolicy(
                    desired_joint_pos=q_initial,
                    kq=kq, kqd=kqd,
            )

    def _setup_mujoco_ik(self):
        self.mujoco_model = MujocoArmModel(self.config)

    def _get_desired_pos_quat(self, eef_pose):
        if self.delta:
            ee_pos_cur, ee_quat_cur = self.robot.get_ee_pose()
            ee_pos_desired = ee_pos_cur + torch.Tensor(eef_pose[:3])

            # add two quaternions
            ee_quat_desired = torch.Tensor(quat_add(ee_quat_cur, eef_pose[3:]))
        else:
            ee_pos_desired = torch.Tensor(eef_pose[:3])
            ee_quat_desired = torch.Tensor(eef_pose[3:])

        return ee_pos_desired, ee_quat_desired
    
    def _apply_joint_commands(self, q_desired):
        q_des_tensor = np.array(q_desired)
        q_des_tensor = torch.tensor(np.clip(
            q_des_tensor, self.JOINT_LIMIT_MIN, self.JOINT_LIMIT_MAX))
        try:
            self.robot.update_current_policy({"q_desired": q_des_tensor.float()})
        except grpc.RpcError:
            self.reset()

    def _apply_eef_commands(self, eef_pose, wait_time=3):

        ee_pos_desired, ee_quat_desired = self._get_desired_pos_quat(eef_pose)

        joint_pos_cur = self.robot.get_joint_positions()
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
        action_obs = {"delta": self.delta}
        if self.action_space == ActionSpace.Cartesian:
            if self.ik_mode == IKMode.Polymetis:
                self._apply_eef_commands(action)

            elif self.ik_mode == IKMode.DMControl:
                ee_pos_current, ee_quat_current = self.robot.get_ee_pose()
                cur_joint_positions = self.robot.get_joint_positions().numpy()
                ee_pos_desired, ee_quat_desired = self._get_desired_pos_quat(action)
                desired_joint_action, _ = self.mujoco_model.local_inverse_kinematics(ee_pos_desired, ee_quat_desired, ee_pos_current, ee_quat_current, cur_joint_positions)
                command_status = self._apply_joint_commands(desired_joint_action)

            action_obs["joint_action"] = desired_joint_action.numpy()
            action_obs["ee_pos_action"] = ee_pos_desired.numpy()
            action_obs["ee_quat_action"] = ee_quat_desired.numpy()
            

        elif self.action_space == ActionSpace.Joint:
            command_status = self._apply_joint_commands(action)
            action_obs["joint_action"] = action
            ee_pos_desired, ee_quat_desired = self.robot.robot_model.forward_kinematics(action)
            action_obs["ee_pos_action"] = ee_pos_desired.numpy()
            action_obs["ee_quat_action"] = ee_quat_desired.numpy()
        return action_obs


    def get_obs(self):
        obs = {}
        joint_positions = self.robot.get_joint_positions()
        eef_position, eef_orientation = self.robot.get_ee_pose()
        obs['q_pos'] = joint_positions.numpy()
        obs['eef_pos'] = eef_position.numpy()
        obs['eef_rot'] = eef_orientation.numpy()
        return obs
