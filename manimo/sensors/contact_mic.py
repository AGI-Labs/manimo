import datetime
import os
import struct
import time
from collections import deque
from multiprocessing import Process, Queue

import numpy as np
import serial
from manimo.sensors import Sensor
from omegaconf import DictConfig

SAMPLE_BLOCK_BYTES = 256 * 2 + 2
NEWLINE = b"\r\n"


def add_audio(audio_frame_queue: Queue, port: str, hz: float):
    """
    Read and audio frames to the multiprocessing Queue
    """
    print("[INFO] Streaming audio")

    ser = serial.Serial(port)
    ser_bytes = ser.read_until(NEWLINE)
    while True:
        try:
            if ser.in_waiting:
                ser_bytes = ser.read(SAMPLE_BLOCK_BYTES)
                ctime = datetime.datetime.now()

                # confirm that full packet was received
                if ser_bytes[-2:] == NEWLINE:
                    # remove newline
                    ser_bytes = ser_bytes[:-2]

                    if len(ser_bytes) == SAMPLE_BLOCK_BYTES - 2:
                        ser_ints = list(struct.iter_unpack("<HHHH", ser_bytes))
                        timestamp = ctime.strftime("%H.%M.%S.%f")
                        audio_frame_queue.put((ser_ints, timestamp))

                else:
                    timestamp = ctime.strftime("%H.%M.%S.%f")
                    print(f"[WARNING] Incomplete packet received at {timestamp}")
                    ser_bytes = ser.read_until(NEWLINE)

                time.sleep(1 / hz)

        except KeyboardInterrupt:
            print("[INFO] Stopping audio stream")
            ser.close()
            break


class ContactMic(Sensor):
    """
    A Sensor interface class for contact mic that provides a gym style
    observation wrapper to contact audio.
    """

    def __init__(self, contact_mic_cfg: DictConfig, baseline_duration: int = 2):
        self.contact_mic_cfg = contact_mic_cfg
        self.name = contact_mic_cfg.name
        self.port = contact_mic_cfg.port

        assert os.path.exists(self.port), (
            f"Contact mic port {self.port} does not exist. "
            "Please check that your sensor is connected to this port and try again."
        )

        self.audio_fps = contact_mic_cfg.audio_fps
        self.audio_packet_size = contact_mic_cfg.audio_packet_size
        self.num_channels = contact_mic_cfg.num_channels
        self.window_dur = contact_mic_cfg.window_dur
        self.baseline_size = baseline_duration * self.audio_fps // self.audio_packet_size
        self.include_timestamp = contact_mic_cfg.include_timestamp

        self.buffer_size = self.window_dur * self.audio_fps // self.audio_packet_size
        self.obs_size = self.window_dur * self.audio_fps

        self.audio_frame_queue = None
        self.observer_proc = None
        self.baseline = None

        self.window = deque(maxlen=self.buffer_size)

        self.start()
        print(f"[INFO] Contact mic initialized on port {self.port}")
        print(f"[INFO] Recording contact audio baseline for {baseline_duration} seconds")
        self._update_baseline()
        print(f"[INFO] Contact mic baseline set")

    def start(self):
        """
        Start the audio stream
        """
        if self.audio_frame_queue is None:
            self.audio_frame_queue = Queue(self.buffer_size)

        if self.observer_proc is None:
            self.observer_proc = Process(
                target=add_audio,
                args=(
                    self.audio_frame_queue,
                    self.port,
                    self.audio_fps,
                ),
            )
            self.observer_proc.start()

    def stop(self):
        """
        Stop the audio stream
        """
        if self.observer_proc is not None:
            self.audio_frame_queue.put(None)
            self.audio_frame_queue.close()
            self.audio_frame_queue.join_thread()
            self.audio_frame_queue = None
            self.observer_proc.join()
            self.observer_proc = None

    def reset(self):
        """
        Reset audio stream, record baseline in default observation
        """
        obs = self.get_obs()
        self.window.clear()

        self._update_baseline()
        info = {"contact_mic_baseline": self.baseline}

        return obs, info

    def get_obs(self) -> dict:
        """
        Maintain sliding window of audio frames and return as observation

        Returns:
            dict: observation dictionary with key as sensor name and value as 
            array of audio frames shape (num_channels, num_samples)
        """
        obs = {self.name: None}
        while not self.audio_frame_queue.empty():
            self.window.append(self.audio_frame_queue.get())

        if len(self.window) > 0:
            if self.include_timestamp:
                obs[self.name] = list(self.window)
            else:
                all_audio = []
                for ser_ints, timestamp in list(self.window):
                    audio_arr = np.array(list(ser_ints), dtype=int).T
                    all_audio.append(audio_arr)

                all_audio_arr = np.concatenate(all_audio, axis=1)

                if all_audio_arr.shape[1] != self.obs_size:
                    # left pad with baseline values
                    baseline_pad = np.full((self.num_channels, self.obs_size - all_audio_arr.shape[1]), self.baseline)
                    all_audio_arr = np.concatenate(
                        (baseline_pad, all_audio_arr), axis=1
                    )
                obs[self.name] = all_audio_arr
        else:
            print("[WARNING] No audio frames in observation")

        return obs

    def _update_baseline(self):
        """
        Update baseline by averaging the first few frames for each channel
        """
        baseline_audio = []
        for _ in range(self.baseline_size):
            ser_ints = self.audio_frame_queue.get()[0]
            audio_arr = np.array(list(ser_ints), dtype=int).T
            baseline_audio.append(audio_arr)
        
        baseline = np.concatenate(baseline_audio, axis=1)
        baseline = np.mean(baseline, axis=1).astype(int)
        self.baseline = baseline.reshape(self.num_channels, 1)
