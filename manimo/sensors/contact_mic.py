import datetime
import os
import struct
import time
from collections import deque
from multiprocessing import Process, Queue

import serial
from manimo.sensors import Sensor
from omegaconf import DictConfig

SAMPLE_BLOCK_BYTES = 256 * 2 + 2
NEWLINE = b"\r\n"


def add_audio(
    contact_mic_cfg: DictConfig, audio_frame_queue: Queue, port: str, hz: float
):
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
    A Sensor interface class for contact mic that provides a gym style observation wrapper to contact audio.
    """

    def __init__(self, contact_mic_cfg: DictConfig):
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

        self.buffer_size = self.window_dur * self.audio_fps // self.audio_packet_size

        self.audio_frame_queue = Queue(self.buffer_size)
        self.observer_proc = None

        self.window = deque(maxlen=self.buffer_size)

        self.start()
        print(f"[INFO] Contact mic initialized on port {self.port}")

    def start(self):
        if self.observer_proc is None:
            self.observer_proc = Process(
                target=add_audio,
                args=(
                    self.contact_mic_cfg,
                    self.audio_frame_queue,
                    self.port,
                    self.audio_fps,
                ),
            )
            self.observer_proc.start()

    def stop(self):
        if self.observer_proc is not None:
            self.observer_proc.terminate()
            self.observer_proc = None

    def reset(self):
        # TODO(@Jared): Add reset code if required or delete this comment
        return self.get_obs(), {}

    def get_obs(self) -> dict:
        obs = {self.name: None}
        while not self.audio_frame_queue.empty():
            self.window.append(self.audio_frame_queue.get())
        if len(self.window) > 0:
            obs[self.name] = list(self.window)
        else:
            print("[WARNING] No audio frames in observation")

        return obs
