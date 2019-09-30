import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time
from collections import deque

from modules.vae_sac_modules import VAESACModule
from modules.lane_tracker import LaneTracker
from utils.utils import create_test_env, get_saved_hyperparams, ALGOS, load_vae

PENALTY_WEIGHT = 0.1
CONTROLS_PER_ACTION = 10


directory_names = {0: "0+day",
                   1: "1+night",
                   2: "day-night-random-agent+",
                   }


class ModuleSelectEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    continuous = False

    def __init__(self, verbose=0, save_log_flag=False, log_num=None, do_proc_simulation=True, custom_num_proc=0):
        super(ModuleSelectEnv, self).__init__()
        self.verbose = verbose
        self.save_log_flag = save_log_flag
        if self.save_log_flag and log_num is not None:
            self._init_log_to_write(log_num)
        self.do_proc_simulation = do_proc_simulation
        self.num_proc = custom_num_proc

        stats_path = "modules/logs/sac/DonkeyVae-v0-level-0_6/DonkeyVae-v0-level-0"
        hyperparams, stats_path = get_saved_hyperparams(stats_path,
                                                    norm_reward=False)
        hyperparams['vae_path'] = ""
        self.inner_env = create_test_env(stats_path=stats_path,
                                        seed=0,
                                        log_dir="modules/logs",
                                        hyperparams=hyperparams)

        day_vae_path = "modules/logs/vae-level-0-dim-32.pkl"
        self.day_vae = load_vae(day_vae_path)
        night_vae_path = "modules/logs_n/vae-32_best.pkl"
        self.night_vae = load_vae(night_vae_path)

        day_model_path = "modules/logs/sac/DonkeyVae-v0-level-0_6/DonkeyVae-v0-level-0.pkl"
        self.day_model = ALGOS["sac"].load(day_model_path)
        night_model_path = "modules/logs_n/sac/DonkeyVae-v0-level-0_1/DonkeyVae-v0-level-0_best.pkl"
        self.night_model = ALGOS["sac"].load(night_model_path)        

        self.num_modules = 2
        self.module0 = VAESACModule(self.inner_env, self.day_model, delay_flag=False)
        self.module1 = VAESACModule(self.inner_env, self.night_model, delay_flag=False)

        if self.continuous:
            # the probability of selection of end-to-end module
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_modules, ))
        else:
            # lane detection, end-to-end
            self.action_space = spaces.Discrete(self.num_modules)

        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(1, 32, ),
                                            dtype=np.float32)

    def step(self, action):
        # TODO:
        ACTION_THRESHOLD = 0.6
        if self.continuous:
            candidates = [i for i, v in enumerate(action) if v >= ACTION_THRESHOLD]
            candidates_value = [v for v in action if v >= ACTION_THRESHOLD]
            if self.previous_action in candidates:
                action = self.previous_action
            else:
                if candidates:
                    candidates_value = softmax(candidates_value)
                    action = int(np.random.choice(candidates, 1, p=candidates_value))
                else:
                    action = softmax(action)
                    action = int(np.random.choice(self.num_modules, 1, p=action))
            self.previous_action = action
        reward_sum = 0
        for _ in range(CONTROLS_PER_ACTION):
            start_time = time.time()
            if action == 0:
                self.num_use[0] += 1
                encoded_obs = self.day_vae.encode(*self.inner_obs)
                self.inner_obs = np.concatenate((encoded_obs, self.inner_env.envs[0].env.command_history), axis=-1)
                inner_action = self.module0.predict(self.inner_obs, self.num_proc)
                check_time(start_time, self.module_response_times)
            elif action == 1:
                self.num_use[1] += 1
                encoded_obs = self.night_vae.encode(*self.inner_obs)
                self.inner_obs = np.concatenate((encoded_obs, self.inner_env.envs[0].env.command_history), axis=-1)
                inner_action = self.module1.predict(self.inner_obs, self.num_proc)
                check_time(start_time, self.module_response_times)
            else:
                print("action error")
            self.inner_obs, reward, done, infos = self.inner_env.step([inner_action])
            if done:
                break

        self.episode_reward += reward_sum
        self.driving_score_percent = np.max((self.inner_env.envs[0].env.viewer.handler.driving_score / 10,
                                             self.driving_score_percent))

        return encoded_obs, reward_sum, done, infos[0]

    def reset(self):
        self.inner_obs = self.inner_env.reset()
        
        if self.verbose == 1:
            self._print_log()
        if self.save_log_flag:
            self._write_log()
        self.driving_score_percent = 0
        self.episode_reward = 0
        self.module_response_times = deque()
        self.num_use = {}
        for i in range(5):
            self.num_use[i] = 0
        self.previous_action = None

        encoded_obs = self.night_vae.encode(*self.inner_obs)
        return encoded_obs

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
        file_name = root_dir + "/result/" + directory_names[simulate_num] + "/"
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
                                  ])
    
    def _write_log(self):
        try:
            ratios = dict_ratio(self.num_use, self.num_modules)
            self.csv_writer.writerow([self.driving_score_percent,
                                      self.episode_reward,
                                      np.mean(self.module_response_times),
                                      np.std(self.module_response_times),
                                      ratios[0], ratios[1],
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
            print("Usage Ratio: {:.2f} {:.2f}".format(ratios[0], ratios[1]))
        except AttributeError:
            pass
        except ZeroDivisionError:
            pass


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
