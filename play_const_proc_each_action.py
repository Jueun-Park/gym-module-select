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
    parser.add_argument('-p', '--num-proc', help='num proc for exp', type=int, default=0)
    args = parser.parse_args()
    return args


args = init_parse_argument()

timestr = time.strftime("%Y%m%d-%H%M%S")
root_dir = os.path.dirname(os.path.abspath((os.path.abspath(__file__))))
file_name = root_dir + "/result/proc-and-reward/"
os.makedirs(file_name, exist_ok=True)
file_name += timestr + ".csv"
print(">>> save csv log file: ", file_name)
csv_file = open(file_name, "a", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
        "action_num",
        "np.average(total_rewards)",
        "np.std(total_rewards)",
        "args.num_exp",
        "exp_num_proc " + str(args.num_proc),
    ])

env = gym.make('ModuleSelect-v1',
               verbose=1,
               do_proc_simulation=False,
               custom_num_proc=args.num_proc,
               )
env = DummyVecEnv([lambda: env])
obs = env.reset()
for action_num in range(5):
    num_done = 0
    total_rewards = []
    try:
        total_reward = 0
        while num_done < args.num_exp:
            action = [action_num]
            obs, rewards, dones, info = env.step(action)
            env.render()
            total_reward += rewards[0]
            if dones[0]:
                num_done += 1
                total_rewards.append(total_reward)
                total_reward = 0
    except KeyboardInterrupt:
        pass

    csv_writer.writerow([
        action_num,
        np.average(total_rewards),
        np.std(total_rewards),
        args.num_exp,
    ])
    csv_file.flush()

env.close()
csv_file.close()
