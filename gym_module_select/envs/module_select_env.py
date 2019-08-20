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

K = 0.5


class ModuleSelectEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    continuous = False

    def __init__(self):
        self.verbose = 1

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
        step_start_time = time.time()
        if self.continuous:
            action = softmax(action)
            action = int(np.random.choice(2, 1, p=action))

        # TODO: do I have to include the time which took when getting the image in processing time?
        self.raw_obs, _, _, _ = self.inner_env.envs[0].env.viewer.observe()
        
        start_time = time.time()
        if action == 0:
            # default line tracer
            self.num_default += 1
            is_done, angle_error = self.detector.detect_lane(self.raw_obs)
            EMERGENCY_MODE = True
            if is_done or EMERGENCY_MODE:
                angle_error = -angle_error
                steer = steer_controller(angle_error)
                reduction = speed_controller(steer)
                speed = base_speed - np.abs(reduction)
            else:
                angle_error = 0
                steer = steer_controller(angle_error)
                speed = base_speed
            inner_action = [[steer, speed]]

            check_processing_time(start_time, self.processing_times)

            self.inner_obs, reward, done, infos = self.inner_env.step(
                inner_action)
        elif action == 1:
            # VAE-SAC agent
            self.num_vae_sac += 1
            inner_action, _ = self.model.predict(
                self.inner_obs, deterministic=True)
            # Clip Action to avoid out of bound errors
            if isinstance(self.inner_env.action_space, gym.spaces.Box):
                inner_action = np.clip(inner_action, self.inner_env.action_space.low,
                                       self.inner_env.action_space.high)
            check_processing_time(start_time, self.processing_times)
            
            self.inner_obs, reward, done, infos = self.inner_env.step(
                inner_action)
        else:
            print("action error")

        cv2.imshow('input', self.detector.original_image_array)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

        time_penalty = np.log(self.processing_times[-1]*50 + 1) * K   # TODO
        reward[0] -= time_penalty
        # print(reward, self.processing_times[-1], time_penalty)
        self.running_reward += reward[0]
        self.ep_len += 1
        check_processing_time(step_start_time, self.step_times)
        return infos[0]['encoded_obs'], reward[0], done, infos[0]

    def reset(self):
        self.inner_obs = self.inner_env.reset()
        self.raw_obs, _, _, _ = self.inner_env.envs[0].env.viewer.observe()
        self._, _, _, infos = self.inner_env.envs[0].env.observe()
        self.detector.detect_lane(self.raw_obs)
        if self.verbose == 1:
            self._print_counting_log()
        self.running_reward = 0
        self.ep_len = 0
        self.processing_times = []
        self.num_default = 0
        self.num_vae_sac = 0
        self.step_times = []
        return infos['encoded_obs']

    def render(self, mode='human', close=False):
        result = self.inner_env.render(mode=mode)
        return result
    
    def close(self):
        self.inner_env.envs[0].env.exit_scene()
        time.sleep(0.5)
        cv2.destroyAllWindows()

    def _print_counting_log(self):
        try:
            print("Episode Reward: {:.2f}".format(self.running_reward))
            print("Episode Length", self.ep_len)
            print("Default:", self.num_default, "/ VAE-SAC:", self.num_vae_sac,
                "/ Default ratio: {:.2f}".format(self.num_default / (self.num_default+self.num_vae_sac)))
            print("One frame processing time mean (ms): {:.2f}".format(
                1000 * np.mean(self.processing_times)))
            print("Step time mean (ms): {:.2f}".format(1000 * np.mean(self.step_times)))
            print("Step per second: {:.2f}".format(1 / np.mean(self.step_times)))
        except ZeroDivisionError:
            pass
        except AttributeError:
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
