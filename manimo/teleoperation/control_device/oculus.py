import numpy as np
import torch

from oculus_reader import OculusReader
from torchcontrol.transform import Rotation as R

from .base import TeleopDeviceReader

def rmat_to_quat(rot_mat, degrees=False):
    quat = R.from_matrix(torch.Tensor(rot_mat)).as_quat()
    return quat

def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X

def get_button_labels(controller_id: str) -> dict:
    if controller_id == "r":
        return {
            "control_en": "rightGrip",
            "grasp_en": "RTr",
        }
    else:
        return {
            "control_en": "leftGrip",
            "grasp_en": "LTr",
        }

class OculusQuestReader(TeleopDeviceReader):
    """Allows for teleoperation using an Oculus controller
    Using the right controller, fully press the grip button (middle finger) to engage teleoperation. Hold B to perform grasp.
    """

    def __init__(self, 
                ip_address,
                lpf_cutoff_hz,
                control_hz,
                controller_id):
        self.reader = OculusReader(ip_address=ip_address) if ip_address is not None else OculusReader()
        self.reader.run()
        self.controller_id = controller_id
        # LPF filter
        self.vr_pose_filtered = None
        self.global_to_env_mat = vec_to_reorder_mat([-3, -1, 2, 4])
        self.vr_to_global_mat = np.eye(4)
        self.reset_orientation = True

        print("Oculus Quest teleop reader instantiated.")

    def get_state(self):
        # Get data from oculus reader
        transforms, buttons = self.reader.get_transformations_and_buttons()

        # Generate output
        button_labels = get_button_labels(self.controller_id)
        if transforms:
            control_en = buttons[button_labels["control_en"]][0] > 0.9
            grasp_en = buttons[button_labels["grasp_en"]]
            # if grasp_en:
            #     print(f"grasp enabled")
            # else:
            #     print(f"grasp not enabled")
            if self.reset_orientation and control_en:
                self.vr_to_global_mat = np.linalg.inv(np.asarray(transforms[self.controller_id]))
                self.reset_orientation = False

            if not control_en:
                self.reset_orientation = True

            diff_matrix = self.vr_to_global_mat @ np.asarray(transforms[self.controller_id])
            # print(f"diff_matrix: {diff_matrix[:3, 3]}")
            pose_matrix = self.global_to_env_mat @ diff_matrix
        else:
            control_en = False
            grasp_en = 0
            pose_matrix = np.eye(4)
            self.vr_pose_filtered = None
            self.reset_orientation = True

        rot_mat = np.asarray(pose_matrix)
        vr_pos = rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])

        pose = (vr_pos, vr_quat)
        return control_en, grasp_en, pose, buttons