import gym
import sys
sys.path.append('.')
import gym_module_select
import os
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

model_name = "/dqn-models/dqn-model_4000.pkl"

env = gym.make('ModuleSelect-v0')
env = DummyVecEnv([lambda: env])
model = DQN.load(os.path.dirname(
    os.path.realpath(__file__)) + model_name)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
