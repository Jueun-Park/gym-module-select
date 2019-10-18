import argparse
import os
import numpy
import gym
import sys
sys.path.append('.')
import time
import gym_module_select
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import SAC

TIMESTEPS = int(1e8) + 1


def init_parse_argument():
    parser = argparse.ArgumentParser()
    timestr = time.strftime("%H%M%S")
    parser.add_argument('-i', '--id', help='nickname of the train', type=str, default=timestr)
    args = parser.parse_args()
    return args


best_mean_reward = -numpy.inf
n_steps = 0
args = init_parse_argument()
datestr = time.strftime("%Y%m%d")
log_directory = os.path.dirname(os.path.realpath(__file__)) + "/sac-log-" + args.id + "-" +  datestr + "/"
model_directory = os.path.dirname(os.path.realpath(__file__)) + "/sac-models-" + args.id + "-" + datestr + "/"


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global best_mean_reward, n_steps
    mean_reward = 0
    if (n_steps + 1) % 100000 == 0:
        print("Saving new best model")
        _locals['self'].save(
            model_directory + 'sac-model_' + str(n_steps + 1) + '.pkl')
    if (n_steps + 1) % 1000 == 0:
        x, y = ts2xy(load_results(log_directory), 'timesteps')
        if len(x) > 0:
            mean_reward = numpy.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                best_mean_reward, mean_reward))

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print("Saving new best model")
            _locals['self'].save(
                model_directory + 'sac-model_' + str(n_steps + 1) + '.pkl')
    n_steps += 1
    return True


if __name__ == "__main__":
    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(model_directory, exist_ok=True)

    env = gym.make('ModuleSelectContinuous-v1',
                    verbose=2,
                    )
    env = Monitor(env, log_directory, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = SAC(
        env=env,
        policy=MlpPolicy,
        verbose=1,
        tensorboard_log="./s1v+1910+sac_tensorboard/",
        batch_size=64,
    )
    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            callback=callback
        )
    except KeyboardInterrupt:
        env.envs[0].env.close()
