import gym
import numpy as np
import time

from modules.gym_vae_donkey.envs.vae_env import DonkeyVAEEnv
from modules.algos import SAC


class VAESACModule:
    def __init__(self, sac_vae_env: DonkeyVAEEnv, model: SAC, delay_weight):
        self.env = sac_vae_env
        self.model = model
        self.delay_weight = delay_weight

    def predict(self, obs, num_proc):
        self._add_delay(num_proc)  # TODO:
        action, _ = self.model.predict(obs, deterministic=True)
        if isinstance(self.env.action_space, gym.spaces.Box):
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

    def _add_delay(self, proc_state):
        # TODO: add delay
        time.sleep(proc_state/10 + self.delay_weight)
