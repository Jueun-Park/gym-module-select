import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import time

from modules.vae_sac_modules import VAESACModule
from utils.utils import create_test_env, get_saved_hyperparams, ALGOS


class ModuleSelectEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    continuous = False

    def __init__(self, log_num=4, custom_delay=0):
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
        self.module1 = VAESACModule(self.inner_env, self.model, 1)
        self.module2 = VAESACModule(self.inner_env, self.model, 2)

        if self.continuous:
            # the probability of selection of end-to-end module
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_modules, ))
        else:
            # lane detection, end-to-end
            self.action_space = spaces.Discrete(self.num_modules)

        # TODO: edit spaces: add simulated num_proc
        self.observation_space = spaces.Box(low=np.finfo(np.float32).min,
                                            high=np.finfo(np.float32).max,
                                            shape=(1, self.inner_env.envs[0].env.z_size),
                                            dtype=np.float32)

    def step(self, action):
        if self.continuous:
            action = softmax(action)
            action = int(np.random.choice(self.num_modules, 1, p=action))
        # TODO:
        if action == 0:
            inner_action = self.module0.predict(self.inner_obs)
        elif action == 1:
            inner_action = self.module1.predict(self.inner_obs)
        elif action == 2:
            inner_action = self.module2.predict(self.inner_obs)
        else:
            print("action error")
        self.inner_obs, reward, done, infos = self.inner_env.step(inner_action)
        return

    def reset(self):
        # TODO:
        self.inner_obs = self.inner_env.reset()

        self._print_log()
        pass

    def render(self, mode='human', close=False):
        result = self.inner_env.render(mode=mode)
        return result
    
    def close(self):
        self.inner_env.envs[0].env.exit_scene()
        time.sleep(0.5)

    def _print_log(self):
        print("=== Reset ===")

    def _simulate_num_proc(self):
        # TODO: make num_proc add term from normal distribution
        pass


class ModuleSelectEnvContinuous(ModuleSelectEnv):
    continuous = True


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
