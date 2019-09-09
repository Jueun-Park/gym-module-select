import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time

from modules.vae_sac_modules import VAESACModule
from utils.utils import create_test_env, get_saved_hyperparams, ALGOS

PENALTY_WEIGHT = 0.5
INIT_NUM_PROC = 0


class ModuleSelectEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    continuous = False

    def __init__(self):
        super(ModuleSelectEnv, self).__init__()

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

        # TODO: add VAESACModules
        self.num_modules = 3
        self.module0 = VAESACModule(self.inner_env, self.model, 0)
        self.module1 = VAESACModule(self.inner_env, self.model, 0.1)
        self.module2 = VAESACModule(self.inner_env, self.model, 0.2)

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
        print(self.num_proc)
        # TODO:
        if self.continuous:
            action = softmax(action)
            action = int(np.random.choice(self.num_modules, 1, p=action))
        self.num_proc = self._simulate_num_proc()
        if action == 0:
            inner_action = self.module0.predict(self.inner_obs, self.num_proc)
        elif action == 1:
            inner_action = self.module1.predict(self.inner_obs, self.num_proc)
        elif action == 2:
            inner_action = self.module2.predict(self.inner_obs, self.num_proc)
        else:
            print("action error")
        self.inner_obs, reward, done, infos = self.inner_env.step(inner_action)
        # TODO: make time penalty term
        time_penalty = 0
        # time_penalty = np.log(self.processing_times[-1]*50 + 1) * PENALTY_WEIGHT
        reward[0] += time_penalty
        self.episode_reward += reward[0]
        self.driving_score_percent = np.max((self.inner_env.envs[0].env.viewer.handler.driving_score / 10,
                                             self.driving_score_percent))
        obs = np.concatenate((infos[0]['encoded_obs'], [[self.num_proc]]), 1)
        return obs, reward, done, infos[0]

    def reset(self):
        # TODO:
        self.inner_obs = self.inner_env.reset()
        self._, _, _, infos = self.inner_env.envs[0].env.observe()
        self._print_log()

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

    def _print_log(self):
        print("=== Reset ===")
        try:
            print("Driving Score (%): {:.2f}".format(self.driving_score_percent))
            print("Episode Reward: {:.2f}".format(self.episode_reward))
        except AttributeError:
            pass

    def _simulate_num_proc(self):
        add_term = np.random.normal(loc=0, scale=1)
        add_term = round(add_term)
        self.num_proc += add_term
        self.num_proc = np.clip(self.num_proc, 0, np.inf)
        return self.num_proc


class ModuleSelectEnvContinuous(ModuleSelectEnv):
    continuous = True


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
