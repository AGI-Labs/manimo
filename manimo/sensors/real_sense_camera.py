from manimo.sensors import Sensor
from multiprocessing import Process, Queue, Value
import numpy as np
from omegaconf import DictConfig
import pyrealsense2 as rs
import time
import copy

def add_image(camera_cfg: DictConfig, rgb_frame_queue: Queue, depth_frame_queue: Queue, hz: float, closed: Value):
    # Configure depth and color streams
    pipe = rs.pipeline()
    config = rs.config()

    config.enable_device(camera_cfg.device_id)
    config.enable_stream(rs.stream.depth, camera_cfg.img_width, camera_cfg.img_height, rs.format.z16, camera_cfg.hz)
    config.enable_stream(rs.stream.color, camera_cfg.img_width, camera_cfg.img_height, rs.format.bgr8, camera_cfg.hz)
    
    pipe.start(config)

    if camera_cfg.warm_start.enabled:
        print(f"[INFO] Warm start cameras (realsense auto-adjusts brightness during initial frames)")
        for _ in range(camera_cfg.warm_start.frames):
            pipe.poll_for_frames()
            time.sleep(1./camera_cfg.hz)

    align = rs.align(rs.stream.color)
    step = 0
    while not closed.value:
        try:
            frames = pipe.poll_for_frames()
            if (frames.is_frameset()):
                    align.process(frames)
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not depth_frame or not color_frame:
                        continue
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_timestamp = depth_frame.get_timestamp()
                    color_image = np.asanyarray(color_frame.get_data())
                    color_timestamp = color_frame.get_timestamp()
                    try:
                        rgb_frame_queue.put((color_image, color_timestamp), block=False)
                        depth_frame_queue.put((depth_image, depth_timestamp), block=False)
                    except:
                        # queue is full, drop the oldest frame
                        try:
                            rgb_frame_queue.get(block=False)
                            depth_frame_queue.get(block=False)
                        except:
                            pass
                            
                    time.sleep(1./hz)
                    step += 1

        except KeyboardInterrupt:
            print("[INFO] Stopping camera stream")
            break

    print("[INFO] Camera stream closed")
                    

class RealSenseCam(Sensor):
    """
    A Sensor interface class for realsense camera that provides a gym style observation
    wrapper to camera images.
    """

    def __init__(self, camera_cfg: DictConfig):
        device_ls = []
        for cam in rs.context().query_devices():
                device_ls.append(cam.get_info(rs.camera_info(1)))
        device_ls.sort()
        self.hz = camera_cfg.hz
        self.device_id = camera_cfg.device_id
        self.name = camera_cfg.name
        assert(self.device_id in device_ls)

        self.rgb_frame_queue = Queue(camera_cfg.buffer_size)
        self.depth_frame_queue = Queue(camera_cfg.buffer_size)
        self.observer_proc = None
        self.camera_cfg = camera_cfg
        self.closed = Value('b', False)

        self.start()
        print(f"[INFO] Camera setup completed.")

    def start(self):
        if self.observer_proc == None:
            self.observer_proc = Process(target=add_image, args=(self.camera_cfg, self.rgb_frame_queue, self.depth_frame_queue, self.hz, self.closed), daemon=True)
            self.observer_proc.start()        

    def close(self):
        """
        Close the image queue and the camera stream process
        """
        if self.observer_proc is not None:
            self.closed.value = True
            self.rgb_frame_queue.close()
            self.rgb_frame_queue.join_thread()
            self.rgb_frame_queue = None
            self.depth_frame_queue.close()
            self.depth_frame_queue.join_thread()
            self.depth_frame_queue = None
            self.observer_proc.join()
            self.observer_proc = None

    def reset(self):
        """
        reset sensor
        """
        # TODO @mohan: implement reset
        obs = self.get_obs()
        info = {}
        return obs, info

    def get_obs(self):
        # TODO: add support for buffersize > 1
        try:
            return {self.name: self.rgb_frame_queue.get()}
        except:
            print(f"{self.name} queue is empty")
            return {self.name: None}
        