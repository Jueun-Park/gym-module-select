import argparse
import gym
import sys
sys.path.append('.')
import gym_module_select
from stable_baselines.common.vec_env import DummyVecEnv


NUM_SIMULATION = 25

def init_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--module', help='module num', type=int, default=0)
    args = parser.parse_args()
    return args


args = init_parse_argument()

env = gym.make('ModuleSelect-v1',
                verbose=1,
                save_log_flag=True,
                log_num=args.module,
                )
env = DummyVecEnv([lambda: env])
num_done = 0
action = [args.module]
try:
    obs = env.reset()
    while num_done < NUM_SIMULATION:
        if args.module == 5:
            action = [env.envs[0].action_space.sample()]
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones[0]:
            num_done += 1
except KeyboardInterrupt:
    pass

env.close()
