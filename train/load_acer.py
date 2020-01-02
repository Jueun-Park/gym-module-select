import gym
import sys
sys.path.append('.')
import gym_module_select
import os
import time
import argparse
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import ACER


def init_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model to load', type=str)
    parser.add_argument('-e', '--num-exp', help='num experiment episode', type=int, default=10)
    parser.add_argument('-l', '--log', help='off logging (default is on)', action='store_false', default=True)
    args = parser.parse_args()
    return args

args = init_parse_argument()
try:
    model_name = os.path.abspath(args.model)
except TypeError:
    print("no model file")

timestr = time.strftime("[%Y%m%d-%H%M%S]")
log_file = open("load_log.txt", 'a', encoding='utf-8')
log_file.write(timestr + args.model + "\n")

env = gym.make('ModuleSelect-v1',
                save_log_flag=args.log,
                log_num=6,
                )
env = DummyVecEnv([lambda: env])
model = ACER.load(model_name)

num_done = 0
try:
    obs = env.reset()
    while num_done < args.num_exp:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones[0]:
            num_done += 1
except KeyboardInterrupt:
    pass

env.close()
