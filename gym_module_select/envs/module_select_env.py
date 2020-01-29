import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time
from collections import deque

from utils.utils import create_test_env, get_saved_hyperparams, ALGOS, load_vae
from modules.vae.controller import VAEController 


LOG_DIR = "/result_dncf_allvae_stacked/"
directory_names = {0: "0+day-clear",
                   1: "1+day-fog",
                   2: "2+night-clear",
                   3: "3+night-fog",
                   4: "random-agent+",
                   5: "dqn-agent+",
                   6: "acer-agent+",
                   }


class ModuleSelectEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    continuous = False

    def __init__(self, verbose=0, save_log_flag=False, log_num=None, controls_per_action=10):
        super(ModuleSelectEnv, self).__init__()
        self.verbose = verbose
        self.save_log_flag = save_log_flag
        if self.save_log_flag and log_num is not None:
            self._init_log_to_write(log_num)
        self.controls_per_action = controls_per_action

        stats_path = "modules/logs_dc_allvae32/sac/DonkeyVae-v0-level-0_1/DonkeyVae-v0-level-0"
        hyperparams, stats_path = get_saved_hyperparams(stats_path,
                                                    norm_reward=False)
        hyperparams['vae_path'] = "modules/logs_dncf/vae-32.pkl"
        self.inner_env = create_test_env(stats_path=stats_path,
                                        seed=0,
                                        log_dir="modules/logs_dncf",
                                        hyperparams=hyperparams)

        dc_module_path = "modules/logs_dc_allvae32/sac/DonkeyVae-v0-level-0_1/DonkeyVae-v0-level-0.zip"
        self.dc_module = ALGOS["sac"].load(dc_module_path)
        df_module_path = "modules/logs_df_allvae32/sac/DonkeyVae-v0-level-0_1/DonkeyVae-v0-level-0.zip"
        self.df_module = ALGOS["sac"].load(df_module_path)
        nc_module_path = "modules/logs_nc_allvae32/sac/DonkeyVae-v0-level-0_1/DonkeyVae-v0-level-0.zip"
        self.nc_module = ALGOS["sac"].load(nc_module_path)
        nf_module_path = "modules/logs_nf_allvae32/sac/DonkeyVae-v0-level-0_1/DonkeyVae-v0-level-0.zip"
        self.nf_module = ALGOS["sac"].load(nf_module_path)

        self.num_modules = 4

        if self.continuous:
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_modules, ))
        else:
            self.action_space = spaces.Discrete(self.num_modules)

        self.n_stacks = 4
        self.latent_size = self.inner_env.envs[0].vae.z_size
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(self.latent_size * self.n_stacks, ),
                                            dtype=np.float32)
        self.obs = np.zeros((self.observation_space.shape))

    def step(self, action):
        if self.continuous:
            action = np.argmax(action)
        reward_sum = 0

        for _ in range(self.controls_per_action):
            start_time = time.time()
            if action == 0:
                self.num_use[0] += 1
                inner_action = self.dc_module.predict(self.inner_obs, deterministic=True)
            elif action == 1:
                self.num_use[1] += 1
                inner_action = self.df_module.predict(self.inner_obs, deterministic=True)
            elif action == 2:
                self.num_use[2] += 1
                inner_action = self.nc_module.predict(self.inner_obs, deterministic=True)
            elif action == 3:
                self.num_use[3] += 1
                inner_action = self.nf_module.predict(self.inner_obs, deterministic=True)
            else:
                print("action error")
            if isinstance(self.inner_env.envs[0].env.action_space, gym.spaces.Box):
                inner_action = np.clip(inner_action[0], self.inner_env.envs[0].env.action_space.low, self.inner_env.envs[0].env.action_space.high)
            check_time(start_time, self.response_times)
            self.inner_obs, reward, done, infos = self.inner_env.step(inner_action)
            reward_sum += reward[0]
            if done:
                break

        self.episode_reward += reward_sum
        self.driving_score_percent = np.max((self.inner_env.envs[0].env.viewer.handler.driving_score / 10,
                                             self.driving_score_percent))
        if done:
            reward_sum += self.driving_score_percent
        self.stack_obs(infos[0]['encoded_obs'])
        return self.obs, reward_sum, done, infos[0]

    def stack_obs(self, new_obs):
        self.obs[self.latent_size:] = self.obs[:self.latent_size * (self.n_stacks - 1)]
        self.obs[:self.latent_size] = new_obs

    def reset(self):
        self.inner_obs = self.inner_env.reset()
        if self.verbose == 1:
            self._print_log()
        if self.save_log_flag:
            self._write_log()
        self.driving_score_percent = 0
        self.episode_reward = 0
        self.response_times = deque()
        self.num_use = {}
        for i in range(5):
            self.num_use[i] = 0
        self.previous_action = None

        self._, _, _, infos = self.inner_env.envs[0].env.observe()
        self.stack_obs(infos['encoded_obs'])
        return self.obs

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
        file_name = root_dir + LOG_DIR + directory_names[simulate_num] + "/"
        os.makedirs(file_name, exist_ok=True)
        file_name += directory_names[simulate_num] + timestr + ".csv"
        print(">>> save csv log file: ", file_name)
        self.csv_file = open(file_name, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["driving score (%)",
                                  "episode reward",
                                  "response time mean",
                                  "usage ratio 0",
                                  "usage ratio 1",
                                  "usage ratio 2",
                                  "usage ratio 3",
                                  ])
    
    def _write_log(self):
        try:
            ratios = dict_ratio(self.num_use, self.num_modules)
            self.csv_writer.writerow([self.driving_score_percent,
                                      self.episode_reward,
                                      np.mean(self.response_times),
                                      ratios[0], ratios[1],
                                      ratios[2], ratios[3],
                                    ])
            self.csv_file.flush()
        except Exception as e:
            print(e)

    def _print_log(self):
        print("=== Reset ===")
        try:
            print("Driving Score (%): {:.2f}".format(self.driving_score_percent))
            print("Episode Reward: {:.2f}".format(self.episode_reward))
            print("Response Time: {:.2f}".format(np.mean(self.response_times)))
            ratios = dict_ratio(self.num_use, self.num_modules)
            print("Usage Ratio: {:.2f} {:.2f} {:.2f} {:.2f}".format(ratios[0], ratios[1], ratios[2], ratios[3]))
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
        if total == 0:
            result.append(0)
        else:
            result.append(dict_in[i] / total)
    return result
