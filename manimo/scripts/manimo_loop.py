import argparse
import hydra
import time
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
            start_time = time.time()
            steps = 0
            for step_idx in range(self.T):
                steps += 1
                action = None
                for callback in self.callbacks:
                    new_action = callback.get_action(obs, pred_action=action)
                    if new_action is not None:
                        action = new_action

                if action is None:
                    time.sleep(0.033)
                    continue
                obs, _, _, _ = self.env.step(action)
                # print(f"env step took: {(time.time() - start_time)/steps}")
                finish = False
                for callback in self.callbacks:
                    finish = callback.on_step(traj_idx, step_idx)
                    if finish:
                        break

                if finish:
                    break
                # print(f"time per step: {(time.time() - start_time) / steps}")

            print(f"fps: {steps / (time.time() - start_time)}")

            for callback in self.callbacks:
                callback.on_end_traj(traj_idx)
            
            # 2. how to exit the entire manimo loop?

            traj_idx += 1

def main():
    manimo_loop = ManimoLoop()


if __name__ == "__main__":
    main()
