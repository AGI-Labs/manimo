from dataclasses import dataclass
from multiprocessing import Process, Queue, Value
import time
from typing import Optional

import cv2
import numpy as np
from omegaconf import DictConfig
import pyzed.sl as sl
from sensor import Sensor

def add_image(camera_cfg, rgb_frame_queue: Queue, closed: Value):

    cam = sl.Camera(cam_cfg.device_id)
    
    init = sl.InitParameters(
				camera_resolution=sl.RESOLUTION.HD720,
				camera_fps=camera_cfg.hz)
    
    cam.open(init)

    runtime = sl.RuntimeParameters()
    runtime.enable_depth = False

    image_left = sl.Mat()
    image_right = sl.Mat()

    step = 0
    try:
        while not closed.value:
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(image_left, sl.VIEW.LEFT)
                cam.retrieve_image(image_right, sl.VIEW.RIGHT)
                color_timestamp = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).data_ns

                color_images = np.array([image_left.get_data()[:, :, :3], 
                                            image_right.get_data()[:, :, :3]], dtype=np.uint8)
            else:
                # raise exception
                raise Exception(f"grabbing image failed with error code: {err}")

            try:
                rgb_frame_queue.put((color_images, color_timestamp), block=False)
            except:
                # queue is full, drop the oldest frame
                try:
                    rgb_frame_queue.get(block=False)
                except:
                    pass
                            
                    time.sleep(1./camera_cfg.hz)
                    step += 1
    except KeyboardInterrupt:
        cam.close()
        print("[INFO] Camera stream closed")

class ZedCam(Sensor):
    """
    A Sensor interface class for the ZED Camera that provides a gym style observation
    wrapper to camera images.
    """
    def __init__(self, cam_cfg: DictConfig):
        self.cam_cfg = cam_cfg
        self.observer_proc = None
        self.closed = Value('b', False)

        # window_dur is the duration of the sliding window in seconds
        self.window_dur = cam_cfg.window_dur
        if self.window_dur is None:
            self.buffer_size = 1  # no sliding window, use single frame at self.hz
            self.window = None
        else:
            self.buffer_size = int(self.window_dur * cam_cfg.hz)
            self.window = deque(maxlen=self.buffer_size)

        self.rgb_frame_queue = Queue(self.buffer_size)

        self.start()

    def start(self):
        if self.observer_proc == None:
            self.observer_proc = Process(target=add_image, args=(self.cam_cfg, self.rgb_frame_queue, self.closed), daemon=True)
            self.observer_proc.start()

    def close(self):
        if self.observer_proc is not None:
            self.closed.value = True
            self.rgb_frame_queue.close()
            self.rgb_frame_queue.join_thread()
            self.rgb_frame_queue = None
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
        name = self.cam_cfg.name
        obs = {name: None}
        try:
            if self.window is None:
                obs[name] = self.rgb_frame_queue.get()
                
            else:
                while not self.rgb_frame_queue.empty():
                    self.window.append(self.rgb_frame_queue.get())

                obs[name] = list(self.window)

        except:
            print(f"{name} queue is empty")
        
        return obs
