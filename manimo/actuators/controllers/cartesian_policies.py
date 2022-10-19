from polymetis.utils.data_dir import get_full_path_to_urdf
import torch
import torchcontrol as toco
from typing import Dict

class CartesianPDPolicy(toco.PolicyModule):
    """
    Performs PD control around a desired cartesian position
    """
    def __init__(self, joint_pos_desired, use_feedforwad, kq, kqd, kx, kxd, **kwargs):
        super().__init__(**kwargs)
        # Get urdf robot model from polymetis
        panda_urdf_path = get_full_path_to_urdf("franka_panda/panda_arm.urdf")
        panda_ee_link_name = "panda_link8"
        self.robot_model = toco.models.RobotModelPinocchio(panda_urdf_path, panda_ee_link_name)
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=True)

        self.joint_pd = toco.modules.feedback.HybridJointSpacePD(kq, kqd, kx, kxd)

        self.use_feedforwad = use_feedforwad
        
        self.joint_pos_desired = torch.nn.Parameter(joint_pos_desired)
        self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)

    def forward(self, state_dict: Dict[str, torch.Tensor]):
        # Parse states
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            self.joint_vel_desired,
            self.robot_model.compute_jacobian(joint_pos_current),
        )

        torque_out = torque_feedback

        if self.use_feedforwad:
            torque_feedforward = self.invdyn(
                joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
            )  # coriolis

            torque_out += torque_feedforward

        return {"joint_torques": torque_out}