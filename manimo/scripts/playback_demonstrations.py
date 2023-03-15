import argparse
import glob
import os
import hydra
import numpy as np
from manimo.environments.single_arm_env import SingleArmEnv
from manimo.utils.helpers import HOMES
from pathlib import Path
import torch


def _separate_filename(filename):
    split = filename.split("_")
    name = "_".join(split[:-1:])
    i = int(split[-1])
    return name, i


def _format_out_dict(list_obs, actions, hz, home):
    out_dict = {k: [] for k in list(list_obs[0].keys())}
    for obs in list_obs:
        for k in out_dict.keys():
            out_dict[k].append(obs[k])
    out_dict = {k: np.array(v) for k, v in out_dict.items()}

    out_dict["actions"] = actions
    out_dict["rate"] = hz
    out_dict["home"] = home
    return out_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    demo_path = Path(args.file)
    name, i = _separate_filename(demo_path.stem)
    num_files = len(glob.glob(f"{demo_path.parents[0]}/{name}_*.npz"))
    data = np.load(args.file)
    home, eef_positions, eef_orientations, hz = (
        data["home"],
        data["eef_pos"],
        data["eef_rot"],
        30,
    )

    hydra.initialize(config_path="../conf", job_name="replay_demos_test")

    actuators_cfg = hydra.compose(config_name="actuators_playback")
    sensors_cfg = hydra.compose(config_name="sensors")

    env = SingleArmEnv(sensors_cfg, actuators_cfg, hz)

    for i in range(i, num_files):
        data = np.load("data/{}_{}.npz".format(name, i))

        user_in = "r"
        while user_in == "r":
            obs = [env.reset()[0]]
            user_in = input("Ready. Loaded {} ({} hz):".format(name, hz))
        actions = []
        home, eef_positions, eef_orientations, hz = (
            data["home"],
            data["eef_pos"],
            data["eef_rot"],
            30,
        )
        joint_traj = data["joint_pos"]
        # Execute trajectory
        for joints in joint_traj:
            # action = torch.Tensor(np.append(eef_positions[j], eef_orientations[j]))
            with torch.no_grad():
                action = torch.Tensor(joints)
                actions.append(action.numpy())
                obs.append(env.step([action])[0])
        env.reset()

        out_dict = _format_out_dict(obs, np.array(actions), hz, home)
        try:
            np.savez("playbacks/{}_{}.npz".format(name, i), **out_dict)
        except FileNotFoundError:
            os.makedirs("playbacks")
            np.savez("playbacks/{}_{}.npz".format(name, i), **out_dict)
        input("Next?")


if __name__ == "__main__":
    main()
