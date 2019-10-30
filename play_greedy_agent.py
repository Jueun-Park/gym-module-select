import argparse
import gym
import gym_module_select
from stable_baselines.common.vec_env import DummyVecEnv
import numpy as np


def init_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num-exp', help='num experiment episode', type=int, default=10)
    args = parser.parse_args()
    return args


args = init_parse_argument()

env = gym.make('ModuleSelect-v1',
               verbose=1,
               save_log_flag=True,
               log_num=6,
               )
env = DummyVecEnv([lambda: env])
num_done = 0


def enqueue(data, queue):
    queue.append(data)
    if len(queue) > 10:  # 10 moving avg
        del queue[0]


rewards = [0]
reward_queue = []
action = [env.envs[0].action_space.sample()]
try:
    obs = env.reset()
    while num_done < args.num_exp:
        if rewards[0] < np.mean(reward_queue):
            action = [env.envs[0].action_space.sample()]
        obs, rewards, dones, info = env.step(action)
        enqueue(rewards[0], reward_queue)
        env.render()
        if dones[0]:
            num_done += 1
except:
    pass

env.close()
