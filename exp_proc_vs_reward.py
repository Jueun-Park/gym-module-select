import argparse
import numpy as np
import os
import csv
import time
import gym
import gym_module_select
from stable_baselines.common.vec_env import DummyVecEnv


def init_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--num-exp', help='num experiment episode', type=int, default=10)
    parser.add_argument('-m', '--max-num-proc', help='max num of processes', type=int, default=10)
    args = parser.parse_args()
    return args


args = init_parse_argument()

timestr = time.strftime("%Y%m%d-%H%M%S")
root_dir = os.path.dirname(os.path.abspath((os.path.abspath(__file__))))
file_name = root_dir + "/result/proc-vs-time/"
os.makedirs(file_name, exist_ok=True)
file_name += timestr + ".csv"
print(">>> save csv log file: ", file_name)
csv_file = open(file_name, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["num_proc",
                     "avg_reward",
                     "reward_std",
                     "(each num proc) num episodes " + str(args.num_exp),
                     ])

for exp_num_proc in range(args.max_num_proc + 1):
    env = gym.make('ModuleSelect-v1',
                verbose=1,
                do_proc_simulation=True,
                custom_num_proc=exp_num_proc,
                )
    env = DummyVecEnv([lambda: env])
    num_done = 0
    total_rewards = []
    try:
        obs = env.reset()
        total_reward = 0
        while num_done < args.num_exp:
            action = [env.envs[0].action_space.sample()]
            obs, rewards, dones, info = env.step(action)
            env.render()
            total_reward += rewards[0]
            if dones[0]:
                num_done += 1
                total_rewards.append(rewards[0])
                total_reward = 0
    except KeyboardInterrupt:
        pass

    env.close()

    csv_writer.writerow([
        exp_num_proc,
        np.average(total_reward),
        np.std(total_reward),
    ])
    csv_file.flush()

csv_file.close()
