import gym
import numpy as np

from modules.gym_vae_donkey.envs.vae_env import DonkeyVAEEnv
from modules.algos import SAC


class VAESACModule:
    def __init__(self, sac_vae_env: DonkeyVAEEnv, model: SAC, delay_weight):
        self.model = model
        self.delay_weight = delay_weight

    def predict(self, obs):
        self._add_delay_from_distribution()
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def _add_delay_from_distribution(self, proc_state):
        # TODO: add delay
        pass
