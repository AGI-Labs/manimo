try:
    from manimo.actuators.arms.franka_arm import FrankaArm
except ImportError as e:    
    print("Failed to import FrankaArm actuator, ", e)