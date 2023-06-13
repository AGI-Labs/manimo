import time

import hydra

hydra.initialize(config_path="../teleoperation/conf", job_name="test_oculus")

teleop_cfg = hydra.compose(config_name="teleop_config")
teleop = hydra.utils.instantiate(teleop_cfg.device)["device"]


while True:
    control_en, grasp_en, vr_pose_cur = teleop.get_state()
    print(f"control_en: {control_en}")
    print(f"grasp_en: {grasp_en}")
    print(f"vr_pose_cur: {vr_pose_cur}")
    time.sleep(1)
