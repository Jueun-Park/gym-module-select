import argparse
import gym
import sys
sys.path.append('.')
import gym_module_select
from stable_baselines.common.vec_env import DummyVecEnv


def init_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--delay', help='custom delay', type=int, default=0)
    args = parser.parse_args()
    return args


args = init_parse_argument()

env = gym.make('ModuleSelect-v0', log_num=0, custom_delay=args.delay)
env = DummyVecEnv([lambda: env])
num_done = 0
try:
    obs = env.reset()
    while num_done < 2:
        action = [0]
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones[0]:
            num_done += 1
except:
    pass

env.close()
