import argparse

import hydra
from manimo.scripts.manimo_loop import ManimoLoop
from manimo.utils.callbacks import BaseCallback
from manimo.teleoperation.teleop_agent import TeleopAgent


class Teleop(BaseCallback):
    """
    Teleoperation callback.
    """

    def __init__(self, logger, teleop_agent):
        super().__init__(logger)
        self.logger = logger
        self.teleop_agent = teleop_agent

    def on_begin_traj(self, traj_idx):
        pass

    def on_end_traj(self, traj_idx):
        pass

    def get_action(self, obs):
        """
        Called at the end of each step.
        """
        arm_action, gripper_action, buttons = self.teleop_agent.get_action(obs)
        if arm_action is not None:
            teleop_action = [arm_action, not gripper_action]
        else:
            teleop_action = None

        print(f"stepping teleop agent")

        return teleop_action

    def on_step(self, traj_idx, step_idx):
        return False


def main():
    teleop_agent = TeleopAgent()
    teleop_callback = Teleop(None, teleop_agent)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    manimo_loop = ManimoLoop(callbacks=[teleop_callback])

    manimo_loop.run()


if __name__ == "__main__":
    main()
