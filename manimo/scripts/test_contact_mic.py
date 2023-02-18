import faulthandler

import hydra
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
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

    if not sensors_cfg['audio']['contact_mic']['contact_mic_cfg']['include_timestamp']:
        # TEST AUDIO OBSERVATIONS
        audio_obs = []
        print(f'collecting audio observations')
        for i in range(1000):
            obs = env.step()[0]
            audio_obs.append(obs["audio"].mean(axis=0))
        env.reset()
        print(f'finished collecting {len(audio_obs)} audio observations')

        def animate(i):
            plt.cla()
            plt.ylim(1900, 2300)
            audio_plot = plt.plot(audio_obs[i])
            return audio_plot

        ani = FuncAnimation(plt.gcf(), animate, frames=len(audio_obs), interval=1000/AUDIO_FPS)
        ani.save("test_audio_obs.gif", writer=PillowWriter(fps=30))
        print("saved audio.gif")

    else:
        # TEST AUDIO WITH TIMESTAMP
        audio_obs = []
        for i in range(100):
            obs = env.step()[0]
            audio_obs += obs["audio"]
        env.reset()
        print(f'finished collecting {len(audio_obs)} audio observations')

        unique_obs = {timestamp: ser_ints for ser_ints, timestamp in audio_obs}
        all_unique_timestamps = list(set([get_sec(t) for t in unique_obs.keys()]))
        all_unique_timestamps.sort()

        print("num unique timestamps: ", len(all_unique_timestamps))
        total_time = all_unique_timestamps[-1] - all_unique_timestamps[0]
        print("total time: ", total_time)

        all_audio_arr = []
        for timestamp, audio_tuple in unique_obs.items():
            audio_arr = np.array(list(audio_tuple), dtype=int).T
            all_audio_arr.append(audio_arr)

        num_audio_blocks = len(all_audio_arr)
        all_audio_arr = np.concatenate(all_audio_arr, axis=1)

        # output timestamp data
        print(f"last timestamp - first timestamp: {total_time} s")
        print(
            f"num_unique_timestamps * PACKET_SIZE / FPS: {len(all_unique_timestamps) * AUDIO_PACKET_SIZE / AUDIO_FPS} s"
        )
        print(
            f"approx fps from timesteps: {len(all_unique_timestamps) * AUDIO_PACKET_SIZE / total_time:.0f} HZ"
        )

        # output audio stats
        print(f"approx fps from audio_arr: {all_audio_arr.shape[1] / total_time:.0f} HZ")
        print("num audio blocks: ", num_audio_blocks)
        print("audio shape: ", all_audio_arr.shape)
        print("audio mean: ", all_audio_arr.mean())
        print("audio std: ", all_audio_arr.std())

        shifted_timesteps = np.array(all_unique_timestamps) - all_unique_timestamps[0]
        shifted_timesteps_full = []
        for i in range(shifted_timesteps.shape[0]):
            shifted_timesteps_full += [shifted_timesteps[i]] * AUDIO_PACKET_SIZE

        # plot audio data
        plt.plot(shifted_timesteps_full, all_audio_arr.mean(axis=0), label="mean")
        for channel in range(all_audio_arr.shape[0]):
            plt.plot(shifted_timesteps_full, all_audio_arr[channel, :], label=f"ch {channel}")
        plt.legend()
        plt.ylim(1900, 2300)
        plt.savefig("./test_audio_all.png")
        print(f'saved test_audio.png')


if __name__ == "__main__":
    main()
