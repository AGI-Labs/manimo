try:
    from manimo.actuators.grippers.franka_gripper import FrankaGripper
except ImportError as e:    
    print("Failed to import FrankaGripper actuator, ", e)