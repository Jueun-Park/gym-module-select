import gym
import sys
sys.path.append('.')
import gym_module_select
from stable_baselines.common.vec_env import DummyVecEnv


env = gym.make('ModuleSelect-v0')
env = DummyVecEnv([lambda: env])

try:
    obs = env.reset()
    for i in range(10000):
        action = [0]
        obs, rewards, dones, info = env.step(action)
        env.render()
except:
    pass

env.close()
