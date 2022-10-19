from manimo.sensors import Sensor
from multiprocessing import Process, Queue
import numpy as np
from omegaconf import DictConfig
import pyrealsense2 as rs
import time

def add_image(pipe, rgb_frame_queue: Queue, depth_frame_queue: Queue, hz: float):
        align = rs.align(rs.stream.color)

        while True:
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
                    rgb_frame_queue.put((color_image, color_timestamp))
                    depth_frame_queue.put((depth_image, depth_timestamp))
                    print(f"getting new frame: {color_timestamp}")
                    # if state.is_logging_to:
                    #     state.cam_recorder_queue.put((state.is_logging_to, i, device_id, color_image, color_timestamp, depth_image, depth_timestamp))


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


        self.pipe = rs.pipeline()
        config = rs.config()

        config.enable_device(self.device_id)
        config.enable_stream(rs.stream.depth, camera_cfg.img_width, camera_cfg.img_height, rs.format.z16, camera_cfg.hz)
        config.enable_stream(rs.stream.color, camera_cfg.img_width, camera_cfg.img_height, rs.format.bgr8, camera_cfg.hz)

        self.pipe.start(config)

        if camera_cfg.warm_start.enabled:
            print(f"[INFO] Warm start cameras (realsense auto-adjusts brightness during initial frames)")
            for _ in range(camera_cfg.warm_start.frames):
                self.pipe.poll_for_frames()
                time.sleep(1./camera_cfg.hz)

        self.rgb_frame_queue = Queue(camera_cfg.buffer_size)
        self.depth_frame_queue = Queue(camera_cfg.buffer_size)
        self.observer_proc = None
        self.start()

        # # Keep polling for frames in a background thread
        # self.cam_state = {}
        # self.pull_thread = Thread(target=update_camera, name="Update cameras",
        #                           args=(self.pipes, self.cam_state, state),
        #                           daemon=True)
        # self.pull_thread.start()

        # self.visual_thread = Thread(target=render_cam_state, name="Render camera states",
        #                           args=[state])
        # self.visual_thread.start()

        print(f"[INFO] Camera setup completed.")

    def start(self):
        if self.observer_proc == None:
            self.observer_proc = Process(target=add_image, args=(self.pipe, self.rgb_frame_queue, self.depth_frame_queue, self.hz))
            self.observer_proc.start()        

    def stop(self):
        self.observer_proc.terminate()

    def get_obs(self):
        return {self.name: self.rgb_frame_queue.get(timeout=1)}
