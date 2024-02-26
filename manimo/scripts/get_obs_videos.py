import argparse
import os
import pickle

import imageio
from robobuf.buffers import ReplayBuffer


def main(in_dir, out_dir, cam_idx):
    # find all the .pkl files in in_dir and store them in a list

    # create out_dir if not exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fnames = []
    for fname in os.listdir(in_dir):
        if fname.endswith(".pkl"):
            fnames.append(fname)

    buffer = ReplayBuffer()

    for fname in sorted(fnames):
        fps = 30  # Specify the desired frames per second
        out_name = f"{fname.split('.')[0]}.mp4"

        with open(f"{in_dir}/{fname}", "rb") as f:
            trajs = pickle.load(f)
            print(f"number of trajectories: {len(trajs)}")
            buffer.append_traj_list(trajs)

        traj_starts = buffer.traj_starts()
        print(f"number of trajectory starts: {len(traj_starts)}")
        all_transitions = 0
        human_transition_count = 0

        for i, traj_start in enumerate(traj_starts):
            transition = traj_start
            writer = imageio.get_writer(f"{out_dir}/{i}_{out_name}", fps=fps)

            while not transition.done:
                if all_transitions == 0:
                    all_transitions += 1
                    continue
                try:
                    if transition.obs.obs["actor"]:
                        human_transition_count += 1
                except KeyError:
                    pass
                all_transitions += 1
                # import pdb; pdb.set_trace()
                img = transition.obs.image(cam_idx)
                # Write the frame to the video file
                writer.append_data(img)
                transition = transition.next
            del transition

            # Release the video writer
            writer.close()

        print(f"number of transitions: {all_transitions}")
        print(f"number of human transitions: {human_transition_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", help="Directory containing .pkl files")
    parser.add_argument("--out_dir", help="Directory to save videos to")
    parser.add_argument(
        "--cam_idx", help="Camera index to save videos from", default=0
    )
    args = parser.parse_args()

    main(args.in_dir, args.out_dir, args.cam_idx)
