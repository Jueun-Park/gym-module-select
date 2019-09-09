import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time

from modules.vae_sac_modules import VAESACModule
from modules.lane_tracker import LaneTracker
from utils.utils import create_test_env, get_saved_hyperparams, ALGOS

PENALTY_WEIGHT = 0.5
INIT_NUM_PROC = 0
CONTROLS_PER_ACTION = 10


directory_names = {0: "0+",
                   1: "1+",
                   2: "2+",
                   3: "3+",
                   4: "4+lane-tracker",
                   }

delay_weights = {0: 0.05,
                 1: 0.1,
                 2: 0.2,
                 3: 0.3,
                 4: 0,
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
        self.module0 = VAESACModule(self.inner_env, self.model, delay_weights[0])
        self.module1 = VAESACModule(self.inner_env, self.model, delay_weights[1])
        self.module2 = VAESACModule(self.inner_env, self.model, delay_weights[2])
        self.module3 = VAESACModule(self.inner_env, self.model, delay_weights[3])
        self.lane_tracker = LaneTracker()

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
            if action == 0:
                inner_action = self.module0.predict(self.inner_obs, self.num_proc)
            elif action == 1:
                inner_action = self.module1.predict(self.inner_obs, self.num_proc)
            elif action == 2:
                inner_action = self.module2.predict(self.inner_obs, self.num_proc)
            elif action == 3:
                inner_action = self.module3.predict(self.inner_obs, self.num_proc)
            elif action == 4:
                inner_action = self.lane_tracker.predict(self.raw_obs)
            else:
                print("action error")
            self.inner_obs, reward, done, infos = self.inner_env.step(inner_action)
            if self.first_flag:
                self.first_flag = False
            else:
                self.raw_obs = infos[0]['raw_obs']
            # TODO: make time penalty term
            time_penalty = 0
            # time_penalty = np.log(self.processing_times[-1]*50 + 1) * PENALTY_WEIGHT
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
        file_name = root_dir + "/result/" + directory_names[simulate_num] + str(delay_weights[simulate_num]) + "/"
        os.makedirs(file_name, exist_ok=True)
        file_name += directory_names[simulate_num] + "-" + timestr + ".csv"
        print(">>> save csv log file: ", file_name)
        self.csv_file = open(file_name, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["driving score (%)",
                                  "episode reward",
                                  ])
    
    def _write_log(self):
        try:
            self.csv_writer.writerow([self.driving_score_percent,
                                      self.episode_reward,
                                    ])
            self.csv_file.flush()
        except:
            pass

    def _print_log(self):
        print("=== Reset ===")
        try:
            print("Driving Score (%): {:.2f}".format(self.driving_score_percent))
            print("Episode Reward: {:.2f}".format(self.episode_reward))
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
