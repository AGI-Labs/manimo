try:
    from manimo.sensors.sensor import Sensor
    from manimo.sensors.real_sense_camera import RealSenseCam
except ImportError as e:    
    print("Failed to import Sensor, ", e)