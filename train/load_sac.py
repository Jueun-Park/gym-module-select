import gym
import sys
sys.path.append('.')
import gym_module_select
import os
import argparse
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC


def init_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model to load', type=str, default='sac-models-190903-td4/sac-model_34000.pkl')
    args = parser.parse_args()
    return args


args = init_parse_argument()
model_name = os.path.abspath(args.model)

env = gym.make('ModuleSelectContinuous-v0')
env = DummyVecEnv([lambda: env])
model = SAC.load(model_name)

obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
