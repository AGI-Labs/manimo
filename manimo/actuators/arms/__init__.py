try:
    from manimo.actuators.arms.kinova_arm import KinovaArm
except ImportError as e:
    print("Failed to import FrankaArm actuator, ", e)
