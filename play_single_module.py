import argparse
import gym
import sys
sys.path.append('.')
import gym_module_select
from stable_baselines.common.vec_env import DummyVecEnv


NUM_SIMULATION = 10

def init_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--module', help='module num', type=int, default=0)
    args = parser.parse_args()
    return args


args = init_parse_argument()

env = gym.make('ModuleSelect-v1')
env = DummyVecEnv([lambda: env])
num_done = 0
# try:
obs = env.reset()
while num_done < NUM_SIMULATION:
    action = [args.module]
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones[0]:
        num_done += 1
# except Exception as e:
#     print("play exception")
#     print(e)

env.close()
