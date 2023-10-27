import argparse

import hydra
from manimo.scripts.eval_loop import Eval
from manimo.scripts.manimo_loop import ManimoLoop
from manimo.scripts.teleop_loop import Teleop
from manimo.teleoperation.teleop_agent import TeleopAgent
from manimo.utils.new_logger import DataLogger
from robobuf.buffers import ReplayBuffer 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./demos")
    parser.add_argument("--agent_paths", nargs="+", default=[])
    args = parser.parse_args()
    replay_buffer = ReplayBuffer()
    logger = DataLogger(replay_buffer=replay_buffer, action_keys=["action"], storage_path=args.log_dir)
    eval_callback = Eval(logger, args.agent_paths)

    teleop_agent = TeleopAgent()
    teleop_callback = Teleop(logger, teleop_agent)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    manimo_loop = ManimoLoop(callbacks=[teleop_callback, eval_callback])
    manimo_loop.run()


if __name__ == "__main__":
    main()
