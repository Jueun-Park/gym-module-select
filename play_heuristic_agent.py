import argparse
import gym
import gym_module_select
from stable_baselines.common.vec_env import DummyVecEnv


def init_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num-exp', help='num experiment episode', type=int, default=10)
    args = parser.parse_args()
    return args


args = init_parse_argument()

env = gym.make('ModuleSelect-v1',
               verbose=1,
               save_log_flag=True,
               log_num=7,
               )
env = DummyVecEnv([lambda: env])
num_done = 0
num_proc = 0
try:
    obs = env.reset()
    while num_done < args.num_exp:
        if 0 <= num_proc <= 1:
            action = [4]
        elif 2 <= num_proc <= 3 or num_proc == 6:
            action = [3]
        elif num_proc == 5:
            action = [2]
        elif 8 <= num_proc <= 10:
            action = [1]
        elif num_proc == 4 or num_proc == 7:
            action = [0]
        obs, rewards, dones, info = env.step(action)
        num_proc = info[0]["num_proc"]
        env.render()
        if dones[0]:
            num_done += 1
except KeyboardInterrupt:
    pass

env.close()
