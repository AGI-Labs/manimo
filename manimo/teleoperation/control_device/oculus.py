import numpy as np
import sophus as sp
import torch

from oculus_reader import OculusReader
from torchcontrol.transform import Rotation as R

from .base import TeleopDeviceReader

def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X

def get_button_labels(controller_id: str) -> dict:
    if controller_id == "R":
        return {
            "control_en": "rightGrip",
            "grasp_en": "B",
        }
    else:
        return {
            "control_en": "leftGrip",
            "grasp_en": "X",
        }

class OculusQuestReader(TeleopDeviceReader):
    """Allows for teleoperation using an Oculus controller
    Using the right controller, fully press the grip button (middle finger) to engage teleoperation. Hold B to perform grasp.
    """

    def __init__(self, ip_address, lpf_cutoff_hz, control_hz, controller_id):
        self.reader = OculusReader(ip_address=ip_address) if ip_address is not None else OculusReader()
        self.reader.run()
        self.controller_id = controller_id
        # LPF filter
        self.vr_pose_filtered = None
        tmp = 2 * np.pi * lpf_cutoff_hz / control_hz
        self.lpf_alpha = tmp / (tmp + 1)
        self.global_to_env_mat = vec_to_reorder_mat([-2, -1, -3, 4])
        self.vr_to_global_mat = np.eye(4)
        self.reset_orientation = True

        print("Oculus Quest teleop reader instantiated.")

    def get_state(self):
        # Get data from oculus reader
        transforms, buttons = self.reader.get_transformations_and_buttons()
        print(f"transforms: {transforms}, buttons: {buttons}")

        # Generate output
        button_labels = get_button_labels(self.controller_id)
        if transforms:
            control_en = buttons[button_labels["control_en"]][0] > 0.9
            grasp_en = buttons[button_labels["grasp_en"]]
            if self.reset_orientation:
                self.vr_to_global_mat = np.linalg.inv(np.asarray(transforms[self.controller_id]))
                self.reset_orientation = False
            pose_matrix = self.global_to_env_mat @ self.vr_to_global_mat @ np.asarray(transforms[self.controller_id])
        else:
            control_en = False
            grasp_en = 0
            pose_matrix = np.eye(4)
            self.vr_pose_filtered = None
            self.reset_orientation = True

        # Create transform (hack to prevent unorthodox matrices)
        r = R.from_matrix(torch.Tensor(pose_matrix[:3, :3]))
        pose_matrix[:3, :3] = sp.SO3.exp(r.as_rotvec()).matrix()
        vr_pose_curr = sp.SE3(pose_matrix)
        # Filter transform
        if self.vr_pose_filtered is None:
            self.vr_pose_filtered = vr_pose_curr
        else:
            self.vr_pose_filtered = self._interpolate_pose(
                self.vr_pose_filtered, vr_pose_curr, self.lpf_alpha
            )
        pose = self.vr_pose_filtered

        return control_en, grasp_en, pose

    @staticmethod
    def _interpolate_pose(pose1, pose2, pct):
        pose_diff = pose1.inverse() * pose2
        return pose1 * sp.SE3.exp(pct * pose_diff.log())