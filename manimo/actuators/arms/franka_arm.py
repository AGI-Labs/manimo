from manimo.actuators.arms.arm import Arm
# from controllers.cartesian_policies import CartesianPDPolicy
import hydra
from omegaconf import DictConfig
from polymetis import RobotInterface
import torch


class FrankaArm(Arm):
    def __init__(self, arm_cfg: DictConfig):
        self.config = arm_cfg
        self.robot = RobotInterface(ip_address=self.config.robot_ip, enforce_version=False)
        # self.policy = CartesianPDPolicy()

        # self.robot.send_torch_policy(self.policy, blocking=False)
    
    def step(self, action):
        self.robot.move_ee_xyz(action)
        # self.robot.move_ee_xyz(torch.Tensor([0, 0, 0.2]))