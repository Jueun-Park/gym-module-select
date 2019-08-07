import gym
from gym import error, spaces, utils
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


class ModuleSelectEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    continuous = False

    def __init__(self):
        stats_path = "logs/sac/DonkeyVae-v0-level-0_6/DonkeyVae-v0-level-0"
        hyperparams, stats_path = get_saved_hyperparams(
            stats_path, norm_reward=False)
        hyperparams['vae_path'] = "logs/vae-level-0-dim-32.pkl"
        self.inner_env = create_test_env(
            stats_path=stats_path, seed=0, log_dir="logs", hyperparams=hyperparams)
        self.inner_obs = self.inner_env.reset()

        self.detector = LaneDetector()
        # TODO: wanna get original image of first frame
        self.raw_obs = np.ones((80, 160, 3), dtype=np.uint8)
        self.detector.detect_lane(self.raw_obs)

        algo = "sac"
        model_path = "logs/sac/DonkeyVae-v0-level-0_6/DonkeyVae-v0-level-0.pkl"
        self.model = ALGOS[algo].load(model_path)

        if self.continuous:
            # the probability of selection of end-to-end module
            self.action_space = spaces.Box(low=-1, high=1, shape=(2, ))
        else:
            # lane detection, end-to-end
            self.action_space = spaces.Discrete(2)
        
        self.observation_space = spaces.Box(low=0, high=255, shape=self.raw_obs.shape)

        self.processing_times = []

    def step(self, action):
        if self.continuous:
            action = softmax(action)
            action = int(np.random.choice(2, 1, p=action))

        start_time = time.time()
        if action == 0:
            # default line tracer
            _, angle_error = self.detector.detect_lane(self.raw_obs)
            angle_error = -angle_error
            steer = steer_controller(angle_error)
            reduction = speed_controller(steer)
            speed = base_speed - np.abs(reduction)
            inner_action = [[steer, speed]]

            self.inner_obs, reward, done, infos = self.inner_env.step(
                inner_action)
        elif action == 1:
            # VAE-SAC agent
            inner_action, _ = self.model.predict(
                self.inner_obs, deterministic=False)
            # Clip Action to avoid out of bound errors
            if isinstance(self.inner_env.action_space, gym.spaces.Box):
                inner_action = np.clip(action, self.inner_env.action_space.low,
                                       self.inner_env.action_space.high)
            self.inner_obs, reward, done, infos = self.inner_env.step(
                [inner_action])
        else:
            print("action error")

        check_processing_time(start_time, self.processing_times)

        self.raw_obs = infos[0]['raw_obs']
        cv2.imshow('input', self.detector.original_image_array)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass

        # TODO: ?
        reward -= self.processing_times[-1]

        return self.raw_obs, reward, done, infos[0]

    def reset(self):
        self.inner_obs = self.inner_env.reset()
        self.raw_obs = np.ones((80, 160, 3), dtype=np.uint8)  # TODO

        return self.raw_obs

    def render(self, mode='human', close=False):
        result = self.inner_env.render(mode=mode)
        return result
    
    def close(self):
        self.inner_env.envs[0].env.exit_scene()
        time.sleep(0.5)
        cv2.destroyAllWindows()


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
