import gym
import sys
sys.path.append('.')
import gym_module_select
import os
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

dir_name = "/sac-models-190903-td4/"
model_name = dir_name + "sac-model_34000.pkl"

env = gym.make('ModuleSelectContinuous-v0')
env = DummyVecEnv([lambda: env])
model = SAC.load(os.path.dirname(
    os.path.realpath(__file__)) + model_name)

obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
