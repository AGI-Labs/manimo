import argparse
import numpy as np
import hydra
from manimo.scripts.manimo_loop import ManimoLoop
from manimo.utils.callbacks import BaseCallback
from manimo.teleoperation.teleop_agent import TeleopAgent
from manimo.utils.new_logger import DataLogger
from robobuf.buffers import ReplayBuffer


class Teleop(BaseCallback):
    """
    Teleoperation callback.
    """

    def __init__(self, logger, teleop_agent):
        super().__init__(logger)
        self.logger = logger
        self.teleop_agent = teleop_agent
        self.buttons = None

    def on_begin_traj(self, traj_idx):
        print(f"beginning teleop for traj {traj_idx}")
        pass

    def on_end_traj(self, traj_idx):
        self.logger.finish(traj_idx)
        pass

    def get_action(self, obs, pred_action=None):
        """
        Called at the end of each step.
        """
        arm_action, gripper_action, buttons = self.teleop_agent.get_action(obs)
        if arm_action is not None:
            teleop_action = [arm_action, not gripper_action]
            new_obs = obs.copy()
            new_obs["action"] = np.append(*teleop_action)
            new_obs["actor"] = "human"
            self.logger.log(new_obs)
            self.buttons = buttons
            print(f"stepping teleop agent")
        else:
            teleop_action = None
        return teleop_action

    def on_step(self, traj_idx, step_idx):
        if self.buttons:
            button_pressed = self.buttons["A"]
            if button_pressed:
                self.buttons = None
                return True
        return False


def main():
    teleop_agent = TeleopAgent()
    replay_buffer = ReplayBuffer()
    logger = DataLogger(replay_buffer=replay_buffer, action_keys=["action"])
    teleop_callback = Teleop(logger, teleop_agent)
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    manimo_loop = ManimoLoop(callbacks=[teleop_callback], T=1000)

    manimo_loop.run()


if __name__ == "__main__":
    main()
