import argparse

import hydra
from manimo.scripts.eval_loop import Eval
from manimo.scripts.manimo_loop import ManimoLoop
from manimo.scripts.teleop_loop import Teleop
from manimo.teleoperation.teleop_agent import TeleopAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_paths", nargs="+", default=[])
    args = parser.parse_args()
    eval_callback = Eval(None, args.agent_paths)

    teleop_agent = TeleopAgent()
    teleop_callback = Teleop(None, teleop_agent)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    manimo_loop = ManimoLoop(callbacks=[eval_callback, teleop_callback])
    manimo_loop.run()


if __name__ == "__main__":
    main()
