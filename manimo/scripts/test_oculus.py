import time

import hydra
from scipy.spatial.transform import Rotation as R

hydra.initialize(config_path="../teleoperation/conf", job_name="test_oculus")

teleop_cfg = hydra.compose(config_name="teleop_config")
teleop = hydra.utils.instantiate(teleop_cfg.device)["device"]

def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler

while True:
    control_en, grasp_en, vr_pose_cur, buttons = teleop.get_state()
    # print(f"control_en: {control_en}")
    # print(f"grasp_en: {grasp_en}")
    # print(f"vr_pose_cur: {vr_pose_cur}")
    try:
        # print(f"vr_pose_cur: {vr_pose_cur}")
        vr_pos, vr_quat = vr_pose_cur
        # convert vr_quat to euler angles
        vr_euler = quat_to_euler(vr_quat, degrees=True)
        print(f"vr_pos: {vr_pos}")
        # print(f"vr_pos[2]: {vr_pos[2]}")
        # print(f"vr_pos: {vr_pos}, vr_euler: {vr_euler}")
    except:
        pass
    time.sleep(0.2)
