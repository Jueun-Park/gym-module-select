import gym
import numpy as np
import time

from modules.gym_vae_donkey.envs.vae_env import DonkeyVAEEnv
from modules.algos import SAC
from modules.add_delay import add_delay

class VAESACModule:
    def __init__(self, sac_vae_env: DonkeyVAEEnv, model: SAC, delay_flag=True, delay_weight=0, static_term=0):
        self.env = sac_vae_env
        self.model = model
        self.delay_weight = delay_weight
        self.static_term = static_term
        self.delay_flag = delay_flag

    def predict(self, obs, num_proc):
        if self.delay_flag:
            add_delay(num_proc, self.delay_weight, self.static_term)
        action, _ = self.model.predict(obs, deterministic=True)
        if isinstance(self.env.action_space, gym.spaces.Box):
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action
