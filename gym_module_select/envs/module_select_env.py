import gym
from gym import error, spaces
from gym.utils import seeding
import numpy as np
import time

import cv2
from simple_pid import PID
from modules.lane_detector import LaneDetector

from gym_module_select.envs.vae_env import DonkeyVAEEnv
from utils.utils import create_test_env, get_saved_hyperparams, ALGOS


steer_controller = PID(Kp=2.88,
                       Ki=0.0,
                       Kd=0.0818,
                       output_limits=(-1, 1),
                       )
base_speed = 1
speed_controller = PID(Kp=1.0,
                       Ki=0.0,
                       Kd=0.125,
                       output_limits=(-1, 1),
                       )

PENALTY_WEIGHT = 0.5
CONTROLS_PER_ACTION = 10
EMERGENCY_MODE = True
# LANE_TRACKER_TIME_DELAY_mu = 0  # 50% chance of delay (negative value is ignored)
# LANE_TRACKER_TIME_DELAY_sigma = 0.05
# TWICE_DELAY = True

directory_names = {0: "0-lane-tracker",
                   1: "1-end-to-end",
                   2: "2-sequence-model",
                   3: "3-orc-model",
                   4: "train-or-test",
                   }
LOG_NUM = 2


class ModuleSelectEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    continuous = False

    def __init__(self):
        self.verbose = 1
        self.save_log_flag = True

        if self.save_log_flag:
            self._init_log_to_write(LOG_NUM)

        stats_path = "logs/sac/DonkeyVae-v0-level-0_6/DonkeyVae-v0-level-0"
        hyperparams, stats_path = get_saved_hyperparams(
            stats_path, norm_reward=False)
        hyperparams['vae_path'] = "logs/vae-level-0-dim-32.pkl"
        self.inner_env = create_test_env(
            stats_path=stats_path, seed=0, log_dir="logs", hyperparams=hyperparams)

        self.detector = LaneDetector()

        algo = "sac"
        model_path = "logs/sac/DonkeyVae-v0-level-0_6/DonkeyVae-v0-level-0.pkl"
        self.model = ALGOS[algo].load(model_path)

        if self.continuous:
            # the probability of selection of end-to-end module
            self.action_space = spaces.Box(low=-1, high=1, shape=(2, ))
        else:
            # lane detection, end-to-end
            self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(1, self.inner_env.envs[0].env.z_size),
                                            dtype=np.float32)

    def step(self, action):
        reward_sum = 0
        if self.continuous:
            action = softmax(action)
            action = int(np.random.choice(2, 1, p=action))
        for i in range(CONTROLS_PER_ACTION):
            start_time = time.time()
            is_done, angle_error = self.detector.detect_lane(self.raw_obs)
            if self.detector.left and self.detector.right:
                # lane tracker
                self.num_lane_tracker += 1
                angle_error = -angle_error
                steer = steer_controller(angle_error)
                reduction = speed_controller(steer)
                speed = base_speed - np.abs(reduction)

                inner_action = [[steer, speed]]

                check_processing_time(start_time, self.processing_times)

                self.inner_obs, reward, done, infos = self.inner_env.step(
                    inner_action)
            else:
                # end-to-end agent
                self.num_end_to_end += 1
                inner_action, _ = self.model.predict(
                    self.inner_obs, deterministic=True)
                # Clip Action to avoid out of bound errors
                if isinstance(self.inner_env.action_space, gym.spaces.Box):
                    inner_action = np.clip(inner_action, self.inner_env.action_space.low,
                                        self.inner_env.action_space.high)
                check_processing_time(start_time, self.processing_times)
                
                self.inner_obs, reward, done, infos = self.inner_env.step(
                    inner_action)

            if self.first_flag:
                self.first_flag = False
            else:
                self.raw_obs = infos[0]['raw_obs']

            cv2.imshow('input', self.detector.original_image_array)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass

            self.original_reward += reward[0]
            time_penalty = np.log(self.processing_times[-1]*50 + 1) * PENALTY_WEIGHT   # TODO
            reward_sum += reward[0] - time_penalty

            self.ep_len += 1
            check_processing_time(start_time, self.step_times)  # check one control time
            if done:
                break

        self.driving_score_percent = np.max((self.inner_env.envs[0].env.viewer.handler.driving_score / 10, self.driving_score_percent))
        self.running_reward += reward_sum
        return infos[0]['encoded_obs'], reward_sum, done, infos[0]

    def reset(self):
        self.inner_obs = self.inner_env.reset()
        self.raw_obs, _, _, _ = self.inner_env.envs[0].env.viewer.observe()  # first observe
        self.first_flag = True
        self._, _, _, infos = self.inner_env.envs[0].env.observe()
        self.detector.detect_lane(self.raw_obs)
        if self.verbose == 1:
            self._print_counting_log()
        if self.save_log_flag:
            self._write_counting_log()
        self.running_reward = 0
        self.original_reward = 0
        self.ep_len = 0
        self.processing_times = []
        self.num_lane_tracker = 0
        self.num_end_to_end = 0
        self.step_times = []
        self.driving_score_percent = 0
        return infos['encoded_obs']

    def render(self, mode='human', close=False):
        result = self.inner_env.render(mode=mode)
        return result
    
    def close(self):
        self.inner_env.envs[0].env.exit_scene()
        time.sleep(0.5)
        cv2.destroyAllWindows()
        self.csv_file.close()

    def _init_log_to_write(self, simulate_num):
        import os
        import csv
        timestr = time.strftime("%Y%m%d-%H%M%S")
        root_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        root_dir = os.path.abspath(os.path.join(root_dir, ".."))
        file_name = root_dir + "/result/" + directory_names[simulate_num] + "/"
        os.makedirs(file_name, exist_ok=True)
        file_name += directory_names[simulate_num] + "-" + timestr + ".csv"
        print(">>> save csv log file: ", file_name)
        self.csv_file = open(file_name, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["original reward",
                                    "driving Score (%)",
                                    "episode reward",
                                    "episode length",
                                    "lane tracker usage ratio",
                                    "one frame processing time mean (ms)",
                                    "one control time mean (ms)",
                                    "controls per second",
                                    "EM mode " + str(EMERGENCY_MODE),
                                    "controls per action " + str(CONTROLS_PER_ACTION),
                                    # "Delay mu " + str(LANE_TRACKER_TIME_DELAY_mu),
                                    # "sigma " + str(LANE_TRACKER_TIME_DELAY_sigma),
                                    # "Twice delay " + str(TWICE_DELAY),
                                    ])

    def _print_counting_log(self):
        print("=== Reset ===")
        try:
            print("Original Reward: {:.2f}".format(self.original_reward))
            print("Driving Score (%): {:.2f}".format(self.driving_score_percent))
            print("Episode Reward: {:.2f}".format(self.running_reward))
            print("Episode Length", self.ep_len)
            print("Lane Tracker:", self.num_lane_tracker,
                  "/ End-to-end Agent:", self.num_end_to_end,
                  "/ Lane tracker usage ratio: {:.2f}".format(
                    self.num_lane_tracker / (self.num_lane_tracker+self.num_end_to_end)))
            print("One frame processing time mean (ms): {:.2f}".format(
                    1000 * np.mean(self.processing_times)))
            print("Step time mean (ms): {:.2f}".format(1000 * np.mean(self.step_times)))
            print("Controls per second: {:.2f}".format(1 / np.mean(self.step_times)))
        except ZeroDivisionError:
            pass
        except AttributeError:
            pass
    
    def _write_counting_log(self):
        try:
            self.csv_writer.writerow([self.original_reward,
                                      self.driving_score_percent,
                                      self.running_reward,
                                      self.ep_len,
                                      self.num_lane_tracker / (self.num_lane_tracker+self.num_end_to_end),
                                      1000 * np.mean(self.processing_times),
                                      1000 * np.mean(self.step_times),
                                      1 / np.mean(self.step_times),
                                      ])
            self.csv_file.flush()
        except:
            # self.csv_writer.writerow(["exception"])
            pass


class ModuleSelectEnvContinuous(ModuleSelectEnv):
    continuous = True


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def check_processing_time(start_time, time_list):
    end_time = time.time()
    processing_time = end_time - start_time
    time_list.append(processing_time)
