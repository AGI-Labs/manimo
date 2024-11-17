import numpy as np
import threading
import time
from oculus_reader import OculusReader
from scipy.spatial.transform import Rotation as R
from collections.abc import Callable
from typing import Any, Tuple

from ..device_base import DeviceBase


def rmat_to_quat(rot_mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion."""
    return R.from_matrix(rot_mat).as_quat()


def vec_to_reorder_mat(vec: np.ndarray) -> np.ndarray:
    """Create reordering matrix from vector."""
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X


def get_button_labels(controller_id: str) -> dict:
    """Get button mapping based on controller ID."""
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


class Se3OculusController(DeviceBase):
    """An Oculus Quest controller for sending SE(3) commands as delta poses for both hands.
    
    This class implements both Oculus Quest controllers to provide commands to robotic arms with grippers.
    It uses the OculusReader interface to communicate with the Oculus Quest headset and controllers.
    
    The command comprises of two parts for each controller:
    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) derived from controller position and orientation
    * gripper: a binary command to open or close the gripper
    
    Control is engaged by fully pressing the grip button (middle finger) on either controller.
    Grasping is controlled using the trigger button on each controller.
    """

    def __init__(self, 
                 ip_address: str = None,
                 lpf_cutoff_hz: float = 20.0,
                 control_hz: float = 100.0,
                 pos_sensitivity: float = 10.0,
                 rot_sensitivity: float = 1.0):
        """Initialize the dual Oculus controller interface.

        Args:
            ip_address: IP address of the Oculus Quest. Defaults to None for auto-discovery.
            lpf_cutoff_hz: Cutoff frequency for low-pass filtering. Defaults to 20.0.
            control_hz: Control loop frequency in Hz. Defaults to 100.0.
            pos_sensitivity: Position scaling factor. Defaults to 1.0.
            rot_sensitivity: Rotation scaling factor. Defaults to 1.0.
        """
        super().__init__()
        
        # Store parameters
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        
        # Initialize OculusReader
        self.reader = OculusReader(ip_address=ip_address) if ip_address else OculusReader()
        self.reader.run()
        
        # Initialize transformation matrices for both controllers
        self.global_to_env_mat = vec_to_reorder_mat([-1, 2, -3, 4])
        self.vr_to_global_mat = {
            'l': np.eye(4),
            'r': np.eye(4)
        }
        self.reset_orientation = {
            'l': True,
            'r': True
        }
        
        # Command buffers for both controllers
        self._close_gripper = {
            'l': False,
            'r': False
        }
        self._delta_pos = {
            'l': np.zeros(3),
            'r': np.zeros(3)
        }
        self._delta_rot = {
            'l': np.zeros(3),
            'r': np.zeros(3)
        }
        
        # Additional callbacks dictionary
        self._additional_callbacks = dict()
        
        # Start update thread
        self._thread = threading.Thread(target=self._run_device)
        self._thread.daemon = True
        self._thread.start()

    def __del__(self):
        """Destructor for the class."""
        self._thread.join()
        
    def __str__(self) -> str:
        """Returns: A string containing the information about both controllers."""
        msg = f"Dual Oculus Quest Controllers for SE(3): {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tLeft and Right Controllers:\n"
        msg += "\tGrip buttons: Enable control for respective arms\n"
        msg += "\tTriggers: Toggle respective gripper commands (open/close)\n"
        msg += "\tMove controllers: Move respective arms in corresponding directions\n"
        msg += "\tRotate controllers: Rotate respective arms correspondingly"
        return msg

    def reset(self) -> None:
        """Reset all commands to default values for both controllers."""
        for controller in ['l', 'r']:
            self._close_gripper[controller] = False
            self._delta_pos[controller] = np.zeros(3)
            self._delta_rot[controller] = np.zeros(3)
            self.reset_orientation[controller] = True
        
    def add_callback(self, key: Any, func: Callable) -> None:
        """Add a callback function for button presses.
        
        Args:
            key: Button identifier ("L" for left, "R" for right)
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        if key not in ["L", "R"]:
            raise ValueError(f"Only left (L) and right (R) buttons supported. Provided: {key}.")
        self._additional_callbacks[key] = func

    def advance(self) -> Tuple[Tuple[np.ndarray, bool], Tuple[np.ndarray, bool]]:
        """Provides the current state for both controllers.
        
        Returns:
            tuple: ((left_delta_pose, left_gripper_command), (right_delta_pose, right_gripper_command))
                - each delta_pose: 6D vector (x, y, z, rx, ry, rz) representing the change in pose
                - each gripper_command: True for close, False for open
        """
        states = {}
        for controller in ['l', 'r']:
            rot_vec = R.from_euler("XYZ", self._delta_rot[controller]).as_rotvec()
            states[controller] = (
                np.concatenate([self._delta_pos[controller], rot_vec]),
                self._close_gripper[controller]
            )
        return states['l'] + states['r']

    def _run_device(self) -> None:
        """Main device update loop for both controllers."""
        while True:
            # Get latest state from Oculus
            transforms, buttons = self.reader.get_transformations_and_buttons()

            if transforms is None:
                print("Warning: No transforms received from Oculus")
                self.reset()  # Reset state when no data
                time.sleep(0.1)  # Shorter sleep when no data
                continue
            
            if transforms:
                # Process both controllers
                for controller_id in ['l', 'r']:
                    # Get button states for current controller
                    button_labels = get_button_labels(controller_id)
                    control_enabled = buttons[button_labels["control_en"]][0] > 0.9
                    grasp_enabled = buttons[button_labels["grasp_en"]]
                    
                    # Handle control enable/disable
                    if self.reset_orientation[controller_id] and control_enabled:
                        self.vr_to_global_mat[controller_id] = np.linalg.inv(
                            np.asarray(transforms[controller_id])
                        )
                        self.reset_orientation[controller_id] = False
                    
                    if not control_enabled:
                        self.reset_orientation[controller_id] = True
                        self._close_gripper[controller_id] = False
                        self._delta_pos[controller_id] = np.zeros(3)
                        self._delta_rot[controller_id] = np.zeros(3)
                        continue

                    # Calculate transformation
                    diff_matrix = self.vr_to_global_mat[controller_id] @ np.asarray(
                        transforms[controller_id]
                    )
                    pose_matrix = self.global_to_env_mat @ diff_matrix
                    
                    # Extract position and rotation
                    self._delta_pos[controller_id] = pose_matrix[:3, 3] * self.pos_sensitivity

                    # Set rotation sensitivity
                    self.rot_sensitivity = 0.05
                    
                    # Convert rotation matrix to Euler angles and apply sensitivity
                    self._delta_rot[controller_id] = R.from_matrix(pose_matrix[:3, :3]).as_euler("XYZ") * self.rot_sensitivity
                    
                    # Zero out rotation along x and z axes
                    self._delta_rot[controller_id][[0, 2]] = 0.0
                    
                    # Update gripper state
                    self._close_gripper[controller_id] = bool(grasp_enabled)
                
            time.sleep(0.0333)  # 30Hz update rate