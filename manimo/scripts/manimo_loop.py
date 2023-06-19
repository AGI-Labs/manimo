import argparse
import hydra

from manimo.environments.single_arm_env import SingleArmEnv


class ManimoLoop:
    def __init__(self, configs=None, callbacks=[], T=2000):
        self.callbacks = callbacks

        if not configs:
            configs = ["sensors", "actuators", "env"]
        
        self.T = T

        hydra.initialize(config_path="../conf", job_name="manimo_loop")

        env_configs = [
            hydra.compose(config_name=config_name) for config_name in configs
        ]

        self.env = SingleArmEnv(*env_configs)


    def run(self):
        traj_idx = 0
        while True:
            obs, _ = self.env.reset()

            for callback in self.callbacks:
                callback.on_begin_traj(traj_idx)

            for step_idx in range(self.T):

                action = None
                for callback in self.callbacks:
                    new_action = callback.get_action(obs)
                    if new_action is not None:
                        action = new_action

                if action is None:
                    continue
                obs, _, _, _ = self.env.step(action)

                finish = False
                for callback in self.callbacks:
                    finish = callback.on_step(traj_idx, step_idx)
                    if finish:
                        break

                if finish:
                    break
    
            for callback in self.callbacks:
                callback.on_end_traj(traj_idx)
            
            # 2. how to exit the entire manimo loop?

            traj_idx += 1

def main():
    manimo_loop = ManimoLoop()


if __name__ == "__main__":
    main()
