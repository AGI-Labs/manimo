try:
    from manimo.actuators.grippers.polymetis_gripper import PolymetisGripper
except ImportError as e:
    print("Failed to import FrankaGripper actuator, ", e)
