from manimo.actuators.arms.arm import Arm
import manimo.actuators.arms.kinova_utils as utilities
from manimo.utils.types import ActionSpace
import numpy as np
from omegaconf import DictConfig
import sys
import os
import threading
import time

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import (
    BaseCyclicClient,
)
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import (
    ControlConfigClient,
)
from kortex_api.autogen.messages import Base_pb2

from kortex_api.autogen.messages import Base_pb2
from manimo.utils.helpers import euler_to_quat

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 10
SLEEP_DURATION = 3

class KinovaArm(Arm):
    def __init__(self, arm_cfg: DictConfig):
        self.config = arm_cfg
        self.action_space = ActionSpace(arm_cfg.action_space)
        self.delta = arm_cfg.delta
        self.hz = arm_cfg.hz
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        # Parse arguments
        self.args = utilities.parseConnectionArguments()

        # Create connection to the device and get the router
        self.connection = utilities.DeviceConnection.createTcpConnection(
            self.args
        )
        self.router = self.connection.__enter__()
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)
        self.control_client = ControlConfigClient(self.router)

        self.prev_gripper_pos = 0
        self.get_obs()

    def _go_home(self, base):
        self.base.Stop()
        time.sleep(SLEEP_DURATION)
        # self.base.StopAction()
        print("Going home through joint commands...")
        action = Base_pb2.Action()
        action.name = "Example angular action movement"
        action.application_data = ""

        actuator_count = base.GetActuatorCount()

        # Place arm straight up
        for joint_id in range(actuator_count.count):
            joint_angle = (
                action.reach_joint_angles.joint_angles.joint_angles.add()
            )
            joint_angle.joint_identifier = joint_id
            joint_angle.value = self.config.home[joint_id]

        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            self._check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )

        print("Executing action")
        base.ExecuteAction(action)

        # open gripper
        gripper_command = self.prepare_gripper_cmd(1)
        self.base.SendGripperCommand(gripper_command)

        print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

        time.sleep(SLEEP_DURATION)

        if finished:
            print("Angular movement completed")
        else:
            print("Timeout on action notification wait")
        return finished

    # Create closure to set an event after an END or an ABORT
    def _check_for_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications

        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """

        def check(notification, e=e):
            print(
                "EVENT : "
                + Base_pb2.ActionEvent.Name(notification.action_event)
            )
            if (
                notification.action_event == Base_pb2.ACTION_END
                or notification.action_event == Base_pb2.ACTION_ABORT
            ):
                e.set()

        return check

    def reset(self):
        print(f"resetting arm")
        self._go_home(self.base)

        return self.obs, {}

    def prepare_gripper_cmd(self, gripper_action):
        gripper_cmd = Base_pb2.GripperCommand()
        gripper_cmd.mode = Base_pb2.GRIPPER_SPEED
        finger = gripper_cmd.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = gripper_action
        return gripper_cmd

    def step(self, next_action):
        action = Base_pb2.Action()
        action.name = "Example Cartesian action movement"
        action.application_data = ""

        dx, dy, dz, dtheta_x, dtheta_y, dtheta_z, gripper = next_action
        GRIPPER_MAX = 1
        gripper = GRIPPER_MAX if gripper == 0 else -GRIPPER_MAX

        delta_max = 1
        delta_theta_max = 1

        dx = np.clip(dx, -delta_max, delta_max)
        dy = np.clip(dy, -delta_max, delta_max)
        dz = np.clip(dz, -delta_max, delta_max)

        dtheta_x = np.clip(dtheta_x, -delta_theta_max, delta_theta_max)
        dtheta_y = np.clip(dtheta_y, -delta_theta_max, delta_theta_max)
        dtheta_z = np.clip(dtheta_z, -delta_theta_max, delta_theta_max)

        command = Base_pb2.TwistCommand()

        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
        command.duration = 0

        twist = command.twist
        twist.linear_x = 0
        twist.linear_y = 0
        twist.linear_z = 0
        twist.linear_x = dy
        twist.linear_y = dx
        twist.linear_z = -dz

        twist.angular_x = dtheta_y
        twist.angular_y = dtheta_x
        twist.angular_z = -dtheta_z

        self.base.SendTwistJoystickCommand(command)

        gripper_command = self.prepare_gripper_cmd(gripper)
        # don't send command if current
        gripper_pos = self.obs["eef_gripper_width"]

        # check if gripper_pos has changed by 0.1 since last time
        # send command only if the gripper has moved last time
        if (
            abs(gripper_pos - self.prev_gripper_pos) < 0.1
            and gripper < 0
            and (not (gripper_pos < 0.1))
        ):
            print(f"not sending gripper command: {gripper_pos}")
        else:
            self.base.SendGripperCommand(gripper_command)

        self.prev_gripper_pos = gripper_pos

        time.sleep(0.033)
        return self.get_obs()

    def get_obs(self):
        obs = {}
        feedback = self.base_cyclic.RefreshFeedback()
        obs["eef_pos"] = np.array(
            [
                feedback.base.tool_pose_x,
                feedback.base.tool_pose_y,
                feedback.base.tool_pose_z,
            ]
        )
        obs["eef_rot"] = euler_to_quat(
            np.array(
                [
                    feedback.base.tool_pose_theta_x,
                    feedback.base.tool_pose_theta_y,
                    feedback.base.tool_pose_theta_z,
                ]
            ),
            degrees=True,
        )

        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION

        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        obs["eef_gripper_width"] = gripper_measure.finger[0].value
        self.obs = obs
        return obs
