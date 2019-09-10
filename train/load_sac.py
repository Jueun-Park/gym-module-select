import gym
import sys
sys.path.append('.')
import gym_module_select
import os
import argparse
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

NUM_SIMULATION = 25


def init_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model to load', type=str, default='train/sac-models-190903-td4/sac-model_34000.pkl')
    args = parser.parse_args()
    return args


args = init_parse_argument()
model_name = os.path.abspath(args.model)

env = gym.make('ModuleSelectContinuous-v1',
                save_log_flag=True,
                log_num=6,
                )
env = DummyVecEnv([lambda: env])
model = SAC.load(model_name)

num_done = 0
try:
    obs = env.reset()
    while num_done < NUM_SIMULATION:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones[0]:
            num_done += 1
except KeyboardInterrupt:
    pass

env.close()
