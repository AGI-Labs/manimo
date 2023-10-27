import argparse
import pickle
import cv2
import os
from robobuf.buffers import ReplayBuffer


def dump_to_file(data, filename):
    # Dump to file
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def main(args):
    fdir = args.directory
    # find all the .pkl files in fdir and store them in a list
    fnames = []
    for fname in os.listdir(fdir):
        if fname.endswith(".pkl"):
            fnames.append(fname)
    traj_count = 0
    if args.human_only:
        skip_actors = ["ai_agent"]
    else:
        skip_actors = []
    buffer = ReplayBuffer()
    num_transitions = 0
    last_buffer_trans = 0
    for fname in sorted(fnames):
        with open(f"{fdir}/{fname}", "rb") as f:
            trajs = pickle.load(f)
            buffer.append_traj_list(trajs, skip_actors=skip_actors)
            last_buffer_trans = len(buffer)

    traj_count = len(buffer.traj_starts())
    traj_list = buffer.to_traj_list()
    out_name = args.output
    dump_to_file(traj_list, out_name)
    print(f"Combined {traj_count} trajectories to obtain {last_buffer_trans} transitions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_only", type=bool, default=False, help="Only store human transitions")
    parser.add_argument("-d", "--directory", type=str, default="/home/nitro/Documents/manimo/manimo/scripts/demos",
                        help="Directory path containing the .pkl files")
    parser.add_argument("-o", "--output", type=str, default="/home/nitro/Documents/manimo/manimo/scripts/traj_list.pkl",
                        help="Output file name")
    args = parser.parse_args()
    main(args)
