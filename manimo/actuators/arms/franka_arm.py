import time

import grpc
import numpy as np
import torch
import torchcontrol as toco
from manimo.actuators.arms.arm import Arm
from manimo.actuators.arms.robot_ik.robot_ik_solver import RobotIKSolver

# from manimo.actuators.arms.moma_arm import MujocoArmModel
from manimo.actuators.controllers.policies import CartesianPDPolicy, JointPDPolicy
from manimo.teleoperation.teleop_agent import quat_add
from manimo.utils.helpers import Rate
from manimo.utils.types import ActionSpace, IKMode
from omegaconf import DictConfig
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation as R


class CustomRobotInterace(RobotInterface):
    def get_ee_pose(self):
        eef_position, eef_orientation = super().get_ee_pose()
        r = toco.transform.Rotation.from_quat(eef_orientation)
        pos_offset = r.apply(
            torch.Tensor([0, 0, 0.145])
        )  # site offset for robotiq pinch
        return eef_position + pos_offset, eef_orientation


def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler


class FrankaArm(Arm):
    def __init__(self, arm_cfg: DictConfig):
        self.config = arm_cfg
        self.action_space = ActionSpace(arm_cfg.action_space)
        self.delta = arm_cfg.delta
        self.hz = arm_cfg.hz
        self.ik_mode = IKMode(arm_cfg.ik_mode)
        self.JOINT_LIMIT_MIN = arm_cfg.joint_limit_min
        self.JOINT_LIMIT_MAX = arm_cfg.joint_limit_max
        self.robot = CustomRobotInterace(
            ip_address=self.config.robot_ip, enforce_version=False
        )
        self.robot.hz = self.hz
        self.kq = arm_cfg.kq
        self.kqd = arm_cfg.kqd
        self.home = (
            arm_cfg.home
            if arm_cfg.home is not None
            else self.robot.get_joint_positions()
        )
        self._ik_solver = RobotIKSolver()
        self.JOINT_OFFSET = np.array(
        [0, 0, 0, 0, 0., np.pi/2, np.pi/4],
        dtype=np.float32)
        self.reset()

    def set_home(self, home):
        self.home = home

    def connect(self, policy=None, wait=2):
        if policy is None:
            policy = self._default_policy(self.action_space)
        self.policy = policy
        self.robot.send_torch_policy(policy, blocking=False)
        time.sleep(wait)

    def get_robot_state(self):
        robot_state = self.robot.get_robot_state()
        gripper_position = 0
        pos, quat = self.robot.robot_model.forward_kinematics(
            torch.Tensor(robot_state.joint_positions)
        )
        cartesian_position = (
            pos.tolist() + quat_to_euler(quat.numpy()).tolist()
        )

        state_dict = {
            "cartesian_position": cartesian_position,
            "gripper_position": gripper_position,
            "joint_positions": list(robot_state.joint_positions),
            "joint_velocities": list(robot_state.joint_velocities),
            "joint_torques_computed": list(robot_state.joint_torques_computed),
            "prev_joint_torques_computed": list(
                robot_state.prev_joint_torques_computed
            ),
            "prev_joint_torques_computed_safened": list(
                robot_state.prev_joint_torques_computed_safened
            ),
            "motor_torques_measured": list(robot_state.motor_torques_measured),
            "prev_controller_latency_ms": (
                robot_state.prev_controller_latency_ms
            ),
            "prev_command_successful": robot_state.prev_command_successful,
        }

        timestamp_dict = {
            "robot_timestamp_seconds": robot_state.timestamp.seconds,
            "robot_timestamp_nanos": robot_state.timestamp.nanos,
        }

        return state_dict, timestamp_dict

    def reset(self):
        self._go_home()
        self.connect()

        obs = self.get_obs()
        return obs, {}

    def _go_home(self):
        home = torch.Tensor(self.home) + torch.Tensor(self.JOINT_OFFSET)

        # Create policy instance
        q_initial = self.robot.get_joint_positions()
        waypoints = toco.planning.generate_joint_space_min_jerk(
            start=q_initial, goal=home, time_to_go=4, hz=self.hz
        )
        rate = Rate(self.hz)
        joint_positions = [waypoint["position"] for waypoint in waypoints]

        q_initial = self.robot.get_joint_positions()
        kq = torch.Tensor(self.robot.metadata.default_Kq)
        kqd = torch.Tensor(self.robot.metadata.default_Kqd)
        policy = JointPDPolicy(
            desired_joint_pos=q_initial,
            kq=kq,
            kqd=kqd,
        )
        self.robot.send_torch_policy(policy, blocking=False)
        rate.sleep()
        for joint_position in joint_positions:
            self.robot.update_current_policy({"q_desired": joint_position})
            rate.sleep()

    def _default_policy(self, action_space, kq_ratio=1.5, kqd_ratio=1.5):
        q_initial = self.robot.get_joint_positions()
        kq = kq_ratio * torch.Tensor(self.kq)
        kqd = kqd_ratio * torch.Tensor(self.kqd)
        kx = torch.Tensor(self.robot.metadata.default_Kx)
        kxd = torch.Tensor(self.robot.metadata.default_Kxd)

        if action_space == ActionSpace.Joint:
            return toco.policies.HybridJointImpedanceControl(
                joint_pos_current=q_initial,
                Kq=kq,
                Kqd=kqd,
                Kx=kx,
                Kxd=kxd,
                robot_model=self.robot.robot_model,
                ignore_gravity=True,
            )
        elif action_space == ActionSpace.Cartesian:
            if self.ik_mode == IKMode.Polymetis:
                return CartesianPDPolicy(q_initial, True, kq, kqd, kx, kxd)
            elif self.ik_mode == IKMode.DMControl:
                return toco.policies.HybridJointImpedanceControl(
                    joint_pos_current=q_initial,
                    Kq=kq,
                    Kqd=kqd,
                    Kx=kx,
                    Kxd=kxd,
                    robot_model=self.robot.robot_model,
                    ignore_gravity=True,
                )
        elif action_space == ActionSpace.JointOnly:
            return toco.policies.JointImpedanceControl(
                joint_pos_current=q_initial,
                Kp=kq,
                Kd=kqd,
                robot_model=self.robot.robot_model,
                ignore_gravity=True,
            )

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

    def _apply_joint_commands(self, q_desired, q_cur):        
        CMD_DELTA_HIGH = np.array([0.1] * 7)/2
        CMD_DELTA_LOW = np.array([-0.1] * 7)/2
        CMD_DELTA_HIGH[-1] *= 7
        CMD_DELTA_LOW[-1] *= 7
        q_des_tensor = torch.tensor(np.clip(q_desired, q_cur+CMD_DELTA_LOW, q_cur+CMD_DELTA_HIGH))
        # q_des_tensor += torch.tensor(self.JOINT_OFFSET)
        q_des_tensor = torch.tensor(
            np.clip(q_des_tensor, self.JOINT_LIMIT_MIN, self.JOINT_LIMIT_MAX)
        )
        try:
            self.robot.update_current_policy(
                {"joint_pos_desired": q_des_tensor.float()}
            )
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
            joint_pos_desired = torch.tensor(
                np.clip(
                    joint_pos_desired,
                    self.JOINT_LIMIT_MIN,
                    self.JOINT_LIMIT_MAX,
                )
            )
            self.robot.update_current_policy(
                {"joint_pos_desired": joint_pos_desired.float()}
            )
        except grpc.RpcError:
            self.robot.send_torch_policy(self.policy, blocking=False)
            update_success = False
            time.sleep(wait_time)

        return update_success

    def step(self, action):
        action_obs = {"delta": self.delta, "action": action.copy()}

        # import pdb; pdb.set_trace()
        if self.action_space == ActionSpace.Cartesian:
            if self.ik_mode == IKMode.Polymetis:
                self._apply_eef_commands(action)

            elif self.ik_mode == IKMode.DMControl:
                ee_pos_current, ee_quat_current = self.robot.get_ee_pose()
                cur_joint_positions = self.robot.get_joint_positions().numpy()
                unscaled_action = action
                # ee_pos_desired, ee_quat_desired = self._get_desired_pos_quat(
                #     unscaled_action
                # )

                robot_state = self.get_robot_state()[0]
                joint_velocity = (
                    self._ik_solver.cartesian_velocity_to_joint_velocity(
                        unscaled_action, robot_state=robot_state
                    )
                )

                joint_delta = self._ik_solver.joint_velocity_to_delta(
                    joint_velocity
                )
                desired_joint_action = (
                    joint_delta + self.robot.get_joint_positions().numpy()
                )
                # desired_joint_action -= self.JOINT_OFFSET
                # cur_joint_positions -= self.JOINT_OFFSET

                command_status = self._apply_joint_commands(
                    desired_joint_action, cur_joint_positions
                )

            action_obs["joint_action"] = desired_joint_action
            # action_obs["ee_pos_action"] = ee_pos_desired.numpy()
            # action_obs["ee_quat_action"] = ee_quat_desired.numpy()

        elif self.action_space == ActionSpace.Joint:
            command_status = self._apply_joint_commands(action)
            action_obs["joint_action"] = action
            (
                ee_pos_desired,
                ee_quat_desired,
            ) = self.robot.robot_model.forward_kinematics(torch.tensor(action))
            action_obs["ee_pos_action"] = ee_pos_desired.numpy()
            action_obs["ee_quat_action"] = ee_quat_desired.numpy()

        elif self.action_space == ActionSpace.JointOnly:
            command_status = self._apply_joint_commands(action)
            action_obs["joint_action"] = action
            (
                ee_pos_desired,
                ee_quat_desired,
            ) = self.robot.robot_model.forward_kinematics(action)
            action_obs["ee_pos_action"] = ee_pos_desired.numpy()
            action_obs["ee_quat_action"] = ee_quat_desired.numpy()
        return action_obs

    def get_obs(self):
        obs = {}
        joint_positions = self.robot.get_joint_positions()
        joint_velocities = self.robot.get_joint_velocities()
        eef_position, eef_orientation = self.robot.get_ee_pose()
        obs["q_pos"] = joint_positions.numpy()
        obs["q_vel"] = joint_velocities.numpy()
        obs["eef_pos"] = eef_position.numpy()
        obs["eef_rot"] = eef_orientation.numpy()
        return obs
