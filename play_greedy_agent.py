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
               log_num=6,
               )
env = DummyVecEnv([lambda: env])
num_done = 0

total_reward = []
rewards = [0]
try:
    obs = env.reset()
    while num_done < args.num_exp:
        if rewards[0] < 2.76:
            action = [env.envs[0].action_space.sample()]
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones[0]:
            num_done += 1
except KeyboardInterrupt:
    pass

env.close()
