import argparse
import gym
import sys
sys.path.append('.')
import gym_module_select
from stable_baselines.common.vec_env import DummyVecEnv


def init_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--module', help='module num', type=int, default=0)
    parser.add_argument('-e', '--num-exp', help='num experiment episode', type=int, default=10)
    args = parser.parse_args()
    return args


args = init_parse_argument()

if args.module == 120:  # 120 means using full daynight model
    action = [120]
    full_flag = True
else:
    action = [args.module]
    full_flag = False

env = gym.make('ModuleSelect-v1',
                verbose=1,
                save_log_flag=True,
                log_num=args.module,
                use_full_daynight_model=full_flag,
                )

env = DummyVecEnv([lambda: env])
num_done = 0
try:
    obs = env.reset()
    while num_done < args.num_exp:
        if args.module == 2:
            action = [env.envs[0].action_space.sample()]
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones[0]:
            num_done += 1
except KeyboardInterrupt:
    pass

env.close()
