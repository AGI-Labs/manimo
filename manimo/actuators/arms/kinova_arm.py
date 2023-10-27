import torch
from manimo.actuators.arms.arm import Arm
import manimo.actuators.arms.kinova_utils as utilities
from manimo.utils.types import ActionSpace, IKMode
import numpy as np
from omegaconf import DictConfig
import sys
import os
import threading
import time

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2
# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 10

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
        self.connection = utilities.DeviceConnection.createTcpConnection(self.args)
        self.router = self.connection.__enter__()
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)

    def _go_home(self, base):
        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(base_servo_mode)
        
        # Move arm to ready position
        print("Moving the arm to a safe position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        if action_handle == None:
            print("Can't reach safe position. Exiting")
            return False

        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            self._check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        base.ExecuteActionFromReference(action_handle)
        # finished = e.wait(TIMEOUT_DURATION)
        finished = True
        base.Unsubscribe(notification_handle)

        if finished:
            print("Safe position reached")
        else:
            print("Timeout on action notification wait")
        return finished

    def _go_new_home(self, base):
        print("Going home through joint commands...")
        action = Base_pb2.Action()
        action.name = "Example angular action movement"
        action.application_data = ""

        actuator_count = base.GetActuatorCount()

        # Place arm straight up
        for joint_id in range(actuator_count.count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = self.config.home[joint_id]

        e = threading.Event()
        notification_handle = base.OnNotificationActionTopic(
            self._check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
        
        print("Executing action")
        base.ExecuteAction(action)

        print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        base.Unsubscribe(notification_handle)

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
        def check(notification, e = e):
            print("EVENT : " + \
                Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    def reset(self):
        print(f"resetting arm")
        # with utilities.DeviceConnection.createTcpConnection(self.args) as router:
        #     self.base = BaseClient(router)
        #     self.base_cyclic = BaseCyclicClient(router)
        # self._go_home(self.base)
        self._go_new_home(self.base)
        
        obs = self.get_obs()
        return obs, {}

    def step(self, next_action):
        # return self.get_obs()
        print("Starting Cartesian action movement ...")
        # with utilities.DeviceConnection.createTcpConnection(self.args) as router:
        #     self.base = BaseClient(router)
        #     self.base_cyclic = BaseCyclicClient(router)
        action = Base_pb2.Action()
        action.name = "Example Cartesian action movement"
        action.application_data = ""

        dx, dy, dz, dtheta_x, dtheta_y, dtheta_z, gripper = next_action
        GRIPPER_MAX = 1.5
        gripper = GRIPPER_MAX if gripper == 0 else -GRIPPER_MAX
        # feedback = self.base_cyclic.RefreshFeedback()

        # cartesian_pose = action.reach_pose.target_pose
        delta_max = 1
        delta_theta_max = 1
        # print(f"dx: {dx} baby")
        
        dx = np.clip(dx, -delta_max, delta_max)
        dy = np.clip(dy, -delta_max, delta_max)
        dz = np.clip(dz, -delta_max, delta_max)

        dtheta_x = np.clip(dtheta_x, -delta_theta_max, delta_theta_max)
        dtheta_y = np.clip(dtheta_y, -delta_theta_max, delta_theta_max)
        dtheta_z = np.clip(dtheta_z, -delta_theta_max, delta_theta_max)

        # print(f"moving in x direction: {delx}")
        print(f"dx: {dx}, dy: {dy}, dz: {dz}")
        command = Base_pb2.TwistCommand()

        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
        command.duration = 0

        twist = command.twist
        twist.linear_x = 0
        twist.linear_y = 0
        twist.linear_z = 0
        # twist.linear_x = -delx
        twist.linear_x = dy
        twist.linear_y = dx
        twist.linear_z = -dz
        
        twist.angular_x = 0
        twist.angular_y = dtheta_x
        twist.angular_z = -dtheta_z
        
        self.base.SendTwistJoystickCommand(command)
        # cartesian_pose.x = feedback.base.tool_pose_x       # (meters)
        # cartesian_pose.y = feedback.base.tool_pose_y    # (meters)
        # cartesian_pose.z = feedback.base.tool_pose_z + delz   # (meters)
        # cartesian_pose.theta_x = feedback.base.tool_pose_theta_x + dtheta_x # (degrees)
        # cartesian_pose.theta_y = feedback.base.tool_pose_theta_y + dtheta_y# (degrees)
        # cartesian_pose.theta_z = feedback.base.tool_pose_theta_z + dtheta_z# (degrees)
                # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Close the gripper with position increments
        print("Performing gripper test in position...")
        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        # position = 0.00
        finger.finger_identifier = 1
        # while position < 1.0:
        finger.value = gripper
            # print("Going to position {:0.2f}...".format(finger.value))
        self.base.SendGripperCommand(gripper_command)
            # position += 0.1
            # time.sleep(1)
        
        
        
        
        
        time.sleep(0.033)

        # e = threading.Event()
        # notification_handle = self.base.OnNotificationActionTopic(
        #     self._check_for_end_or_abort(e),
        #     Base_pb2.NotificationOptions()
        # )

        # print("Executing action")
        # self.base.ExecuteAction(action)

        # print("Waiting for movement to finish ...")
        # # finished = True
        # # finished = e.wait(0.3)
        # # self.base.Unsubscribe(notification_handle)

        # if finished:
        #     print("Cartesian movement completed")
        # else:
        #     print("Timeout on action notification wait")
        return self.get_obs()

    def get_obs(self):
        obs = {}
        # with utilities.DeviceConnection.createTcpConnection(self.args) as router:
        #     self.base = BaseClient(router)
        #     self.base_cyclic = BaseCyclicClient(router)
        feedback = self.base_cyclic.RefreshFeedback()
        obs["eef_pos"] = np.array([feedback.base.tool_pose_x,
                                feedback.base.tool_pose_y,
                                feedback.base.tool_pose_z])
        obs["eef_rot"] = np.array([feedback.base.tool_pose_theta_x,
                                feedback.base.tool_pose_theta_y,
                                feedback.base.tool_pose_theta_z])
        # joint_positions = self.robot.get_joint_positions()
        # joint_velocities = self.robot.get_joint_velocities()
        # eef_position, eef_orientation = self.robot.get_ee_pose()
        # obs["q_pos"] = joint_positions.numpy()
        # obs["q_vel"] = joint_velocities.numpy()
        # obs["eef_pos"] = eef_position.numpy()
        # obs["eef_rot"] = eef_orientation.numpy()
        return obs
