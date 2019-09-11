import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time
from collections import deque

from modules.vae_sac_modules import VAESACModule
from modules.lane_tracker import LaneTracker
from utils.utils import create_test_env, get_saved_hyperparams, ALGOS

PENALTY_WEIGHT = 0.1
INIT_NUM_PROC = 0
CONTROLS_PER_ACTION = 10


directory_names = {0: "0+",
                   1: "1+",
                   2: "2+",
                   3: "3+",
                   4: "4+",
                   5: "random-agent",
                   6: "sac-agent",
                   }

"""
d ~ Exp(1/num_proc(ms))
t = d * weight
bigger w, bigger std of response time
std term
"""
delay_weights = {0: 0.05,
                 1: 0.1,
                 2: 0.15,
                 3: 0.2,
                 4: 0.35,
                 5: "",
                 6: "",
                 }
"""
wait_time = t + static_term
bigger add term, bigger mean of response time
mean term
"""
static_terms = {0: 0.07,
                1: 0.065,
                2: 0.06,
                3: 0.055,
                4: 0.05,
                5: "",
                6: "",
                }

class ModuleSelectEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    continuous = False

    def __init__(self, verbose=0, save_log_flag=False, log_num=None):
        super(ModuleSelectEnv, self).__init__()
        self.verbose = verbose
        self.save_log_flag = save_log_flag
        if self.save_log_flag and log_num is not None:
            self._init_log_to_write(log_num)

        stats_path = "modules/logs/sac/DonkeyVae-v0-level-0_6/DonkeyVae-v0-level-0"
        hyperparams, stats_path = get_saved_hyperparams(stats_path,
                                                    norm_reward=False)
        hyperparams['vae_path'] = "modules/logs/vae-level-0-dim-32.pkl"
        self.inner_env = create_test_env(stats_path=stats_path,
                                        seed=0,
                                        log_dir="modules/logs",
                                        hyperparams=hyperparams)

        model_path = "modules/logs/sac/DonkeyVae-v0-level-0_6/DonkeyVae-v0-level-0.pkl"
        self.model = ALGOS["sac"].load(model_path)

        self.num_modules = 5
        self.module0 = VAESACModule(self.inner_env, self.model, delay_weights[0], static_terms[0])
        self.module1 = VAESACModule(self.inner_env, self.model, delay_weights[1], static_terms[1])
        self.module2 = VAESACModule(self.inner_env, self.model, delay_weights[2], static_terms[2])
        self.module3 = VAESACModule(self.inner_env, self.model, delay_weights[3], static_terms[3])
        self.module4 = VAESACModule(self.inner_env, self.model, delay_weights[4], static_terms[4])

        if self.continuous:
            # the probability of selection of end-to-end module
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_modules, ))
        else:
            # lane detection, end-to-end
            self.action_space = spaces.Discrete(self.num_modules)

        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(1, self.inner_env.envs[0].env.z_size + 1, ),
                                            dtype=np.float32)

    def step(self, action):
        # TODO:
        if self.continuous:
            action = softmax(action)
            action = int(np.random.choice(self.num_modules, 1, p=action))
        reward_sum = 0
        self.num_proc = self._simulate_num_proc()
        for _ in range(CONTROLS_PER_ACTION):
            start_time = time.time()
            if action == 0:
                self.num_use[0] += 1
                inner_action = self.module0.predict(self.inner_obs, self.num_proc)
                check_time(start_time, self.module_response_times)
            elif action == 1:
                self.num_use[1] += 1
                inner_action = self.module1.predict(self.inner_obs, self.num_proc)
                check_time(start_time, self.module_response_times)
            elif action == 2:
                self.num_use[2] += 1
                inner_action = self.module2.predict(self.inner_obs, self.num_proc)
                check_time(start_time, self.module_response_times)
            elif action == 3:
                self.num_use[3] += 1
                inner_action = self.module3.predict(self.inner_obs, self.num_proc)
                check_time(start_time, self.module_response_times)
            elif action == 4:
                self.num_use[4] += 1                
                inner_action = self.module4.predict(self.inner_obs, self.num_proc)
                check_time(start_time, self.module_response_times)
            else:
                print("action error")
            self.inner_obs, reward, done, infos = self.inner_env.step(inner_action)
            if self.first_flag:
                self.first_flag = False
            else:
                self.raw_obs = infos[0]['raw_obs']
            # time_penalty = 0
            time_penalty = np.log(self.module_response_times[-1]*50 + 1) * PENALTY_WEIGHT
            time_penalty = np.clip(time_penalty, 0, reward[0])
            reward_sum += reward[0] - time_penalty
            if done:
                break

        self.episode_reward += reward_sum
        self.driving_score_percent = np.max((self.inner_env.envs[0].env.viewer.handler.driving_score / 10,
                                             self.driving_score_percent))
        obs = np.concatenate((infos[0]['encoded_obs'], [[self.num_proc]]), 1)
        return obs, reward_sum, done, infos[0]

    def reset(self):
        self.inner_obs = self.inner_env.reset()
        self._, _, _, infos = self.inner_env.envs[0].env.observe()
        self.raw_obs, _, _, _ = self.inner_env.envs[0].env.viewer.observe()  # first observe
        self.first_flag = True
        
        if self.verbose == 1:
            self._print_log()
        if self.save_log_flag:
            self._write_log()

        self.num_proc = INIT_NUM_PROC
        self.driving_score_percent = 0
        self.episode_reward = 0
        self.module_response_times = deque()
        self.num_use = {}
        for i in range(5):
            self.num_use[i] = 0

        obs = np.concatenate((infos['encoded_obs'], [[self.num_proc]]), 1)
        return obs

    def render(self, mode='human', close=False):
        result = self.inner_env.render(mode=mode)
        return result
    
    def close(self):
        self.inner_env.envs[0].env.exit_scene()
        time.sleep(0.5)

    def _init_log_to_write(self, simulate_num):
        import os
        import csv
        timestr = time.strftime("%Y%m%d-%H%M%S")
        root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        root_dir = os.path.abspath(os.path.join(root_dir, ".."))
        file_name = root_dir + "/result/" + directory_names[simulate_num] + \
                                            str(delay_weights[simulate_num]) + "+" + \
                                            str(static_terms[simulate_num]) + "/"
        os.makedirs(file_name, exist_ok=True)
        file_name += directory_names[simulate_num] + timestr + ".csv"
        print(">>> save csv log file: ", file_name)
        self.csv_file = open(file_name, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["driving score (%)",
                                  "episode reward",
                                  "response time mean",
                                  "response time std",
                                  "usage ratio 0",
                                  "usage ratio 1",
                                  "usage ratio 2",
                                  "usage ratio 3",
                                  "usage ratio 4",
                                  str(delay_weights[simulate_num]),
                                  str(static_terms[simulate_num]),
                                  ])
    
    def _write_log(self):
        try:
            ratios = dict_ratio(self.num_use, self.num_modules)
            self.csv_writer.writerow([self.driving_score_percent,
                                      self.episode_reward,
                                      np.mean(self.module_response_times),
                                      np.std(self.module_response_times),
                                      ratios[0], ratios[1], ratios[2], ratios[3], ratios[4],
                                    ])
            self.csv_file.flush()
        except:
            pass

    def _print_log(self):
        print("=== Reset ===")
        try:
            print("Driving Score (%): {:.2f}".format(self.driving_score_percent))
            print("Episode Reward: {:.2f}".format(self.episode_reward))
            print("Response Time Mean: {:.2f}".format(np.mean(self.module_response_times)))
            print("Response Time std: {:.2f}".format(np.std(self.module_response_times)))
            ratios = dict_ratio(self.num_use, self.num_modules)
            print("Usage Ratio: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                ratios[0], ratios[1], ratios[2], ratios[3], ratios[4]))
        except AttributeError:
            pass

    def _simulate_num_proc(self):
        add_term = np.random.normal(loc=0, scale=0.9)
        add_term = round(add_term)
        self.num_proc += add_term
        self.num_proc = np.clip(self.num_proc, 0, np.inf)
        return int(self.num_proc)


class ModuleSelectEnvContinuous(ModuleSelectEnv):
    continuous = True


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def check_time(start_time, time_deque: deque):
    end_time = time.time()
    processing_time = end_time - start_time
    time_deque.append(processing_time * 1000)  # ms

def dict_ratio(dict_in, num):
    total = 0
    for i in range(num):
        total += dict_in[i]
    result = []
    for i in range(num):
        result.append(dict_in[i] / total)
    return result
