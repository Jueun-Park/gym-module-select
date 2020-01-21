import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time
from collections import deque

from utils.utils import create_test_env, get_saved_hyperparams, ALGOS, load_vae
from modules.vae.controller import VAEController 

CONTROLS_PER_ACTION = 1

LOG_DIR = "/result_dncf_allvae/"
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

    def __init__(self, verbose=0, save_log_flag=False, log_num=None, use_full_daynight_model=False):
        super(ModuleSelectEnv, self).__init__()
        self.verbose = verbose
        self.save_log_flag = save_log_flag
        if self.save_log_flag and log_num is not None:
            self._init_log_to_write(log_num)
        self.use_full_daynight_model = use_full_daynight_model

        stats_path = "modules/logs_dc/sac/DonkeyVae-v0-level-0_1/DonkeyVae-v0-level-0"
        hyperparams, stats_path = get_saved_hyperparams(stats_path,
                                                    norm_reward=False)
        hyperparams['vae_path'] = "modules/logs_dc/vae-32.pkl"
        self.inner_env = create_test_env(stats_path=stats_path,
                                        seed=0,
                                        log_dir="modules/logs_dc",
                                        hyperparams=hyperparams)

        if self.use_full_daynight_model:
            # TODO: not yet in dncf
            # the model trained in discrete light changing env
            full_vae_path = "modules/logs_daynight_full/vae-32.pkl"
            self.full_vae = load_vae(full_vae_path)
            full_model_path = "modules/logs_daynight_full/sac/DonkeyVae-v0-level-0_1/DonkeyVae-v0-level-0_best.pkl"
            self.full_model = ALGOS["sac"].load(full_model_path)
        else:
            dc_vae_path = "modules/logs_dc/vae-32.pkl"
            self.dc_vae = load_vae(dc_vae_path)
            df_vae_path = "modules/logs_df01/vae-32.pkl"
            self.df_vae = load_vae(df_vae_path)
            nc_vae_path = "modules/logs_nc/vae-32.pkl"
            self.nc_vae = load_vae(nc_vae_path)
            nf_vae_path = "modules/logs_nf01/vae-32.pkl"
            self.nf_vae = load_vae(nf_vae_path)

            dc_module_path = "modules/logs_dc/sac/DonkeyVae-v0-level-0_1/DonkeyVae-v0-level-0_best_63374.zip"  # ? 55-99%
            self.dc_module = ALGOS["sac"].load(dc_module_path)
            df_module_path = "modules/logs_df01/sac/DonkeyVae-v0-level-0_1/DonkeyVae-v0-level-0_best_62409.zip"  # ? 15-16%
            self.df_module = ALGOS["sac"].load(df_module_path)
            nc_module_path = "modules/logs_nc/sac/DonkeyVae-v0-level-0_1/DonkeyVae-v0-level-0_best_61638.zip"  # ? 30-97%
            self.nc_module = ALGOS["sac"].load(nc_module_path)
            nf_module_path = "modules/logs_nf01/sac/DonkeyVae-v0-level-0_1/DonkeyVae-v0-level-0_best_62461.zip"  # ? 31-61%
            self.nf_module = ALGOS["sac"].load(nf_module_path)

        self.num_modules = 4

        if self.continuous:
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_modules, ))
        else:
            self.action_space = spaces.Discrete(self.num_modules)

        self.all_vae = VAEController()
        self.all_vae.load("modules/logs_dncf/vae-32.pkl")

        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(self.all_vae.z_size, ),
                                            dtype=np.float32)

    def step(self, action):
        if self.continuous:
            action = np.argmax(action)
        reward_sum = 0

        if action == 0:
            self.inner_env.envs[0].set_vae(self.dc_vae)
        elif action == 1:
            self.inner_env.envs[0].set_vae(self.df_vae)
        elif action == 2:
            self.inner_env.envs[0].set_vae(self.nc_vae)
        elif action == 3:
            self.inner_env.envs[0].set_vae(self.nf_vae)
        elif self.use_full_daynight_model:
            self.inner_env.envs[0].set_vae(self.full_vae)

        for _ in range(CONTROLS_PER_ACTION):
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
            elif self.use_full_daynight_model:
                # TODO: full model
                pass
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
        return self.all_vae.encode(infos[0]['raw_image']), reward_sum, done, infos[0]

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
        if self.use_full_daynight_model:
            self.inner_env.envs[0].set_vae(self.full_vae)
        else:
            self.inner_env.envs[0].set_vae(self.dc_vae)
        return self.all_vae.encode(infos['raw_image'])

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
