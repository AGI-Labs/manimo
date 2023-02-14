import faulthandler

import hydra
import matplotlib.pyplot as plt
import numpy as np
from manimo.environments.single_arm_env import SingleArmEnv

AUDIO_PACKET_SIZE = 64
AUDIO_FPS = 32000


def get_sec(time_str):
    """Get Seconds from time."""
    hour, minute, second, second_decimal = time_str.split(".")
    return (
        int(hour) * 3600 + int(minute) * 60 + int(second) + float("0." + second_decimal)
    )


def main():
    # create a single arm environment
    hydra.initialize(config_path="../conf", job_name="collect_demos_test")

    # configure actuators and sensors
    actuators_cfg = hydra.compose(config_name="actuators")
    sensors_cfg = hydra.compose(config_name="sensors")

    # create environment
    faulthandler.enable()
    env = SingleArmEnv(sensors_cfg, actuators_cfg)

    audio_obs = []
    for i in range(100):
        obs = env.get_obs()
        audio_obs += obs["audio"]
    env.reset()

    all_unique_timestamps = list(set([get_sec(t) for _, t in audio_obs]))
    all_unique_timestamps.sort()
    print("num unique timestamps: ", len(all_unique_timestamps))
    total_time = all_unique_timestamps[-1] - all_unique_timestamps[0]
    print("total time: ", total_time)

    # process audio
    all_audio_arr = []
    all_timestamps = set()
    for audio_tuple, timestamp in audio_obs:
        # add timestamp to set
        if timestamp not in all_timestamps:
            all_timestamps.add(timestamp)
            audio_arr = np.array(list(audio_tuple), dtype=np.float64).T
            all_audio_arr.append(audio_arr)

    num_audio_blocks = len(all_audio_arr)
    num_timestamps = len(all_timestamps)
    all_audio_arr = np.concatenate(all_audio_arr, axis=1)

    # output timestamp data
    print(f"last timestamp - first timestamp: {total_time}")
    print(
        f"num_unique_timestamps * PACKET_SIZE / FPS: {len(all_unique_timestamps) * AUDIO_PACKET_SIZE / AUDIO_FPS}"
    )
    print(
        f"approx fps from timesteps: {len(all_unique_timestamps) * AUDIO_PACKET_SIZE / total_time:.0f} HZ"
    )

    # output audio stats
    print(f"approx fps from audio_arr: {all_audio_arr.shape[1] / total_time:.0f} HZ")
    print("num audio blocks: ", num_audio_blocks)
    print("num timestamps: ", num_timestamps)
    print("audio shape: ", all_audio_arr.shape)
    print("audio mean: ", all_audio_arr.mean())
    print("audio std: ", all_audio_arr.std())

    shifted_timesteps = np.array(all_unique_timestamps) - all_unique_timestamps[0]
    shifted_timesteps_full = []
    for i in range(shifted_timesteps.shape[0]):
        shifted_timesteps_full += [shifted_timesteps[i]] * AUDIO_PACKET_SIZE

    # plot audio data
    plt.plot(shifted_timesteps_full, all_audio_arr.mean(axis=0))
    # for channel in range(all_audio_arr.shape[0]):
    #     plt.plot(shifted_timesteps_full, all_audio_arr[channel, :])
    plt.savefig("./test_audio.png")


if __name__ == "__main__":
    main()
