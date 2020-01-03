import argparse
import os
import numpy
import gym
import sys
sys.path.append('.')
import time
import gym_module_select
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import ACER


def init_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--id', help='nickname of the train', type=str, default="")
    parser.add_argument('-l', '--learning-rate', help='learning rate', type=float, default=7e-4)
    parser.add_argument('-t', '--timesteps', help='timesteps', type=int, default=100000001)
    args = parser.parse_args()
    return args


best_mean_reward = -numpy.inf
n_steps = 0
args = init_parse_argument()
datestr = time.strftime("%Y%m%d")
timestr = time.strftime("%H%M%S")
log_directory = os.path.dirname(os.path.realpath(__file__)) + "/" + datestr + "-dncf-acer-log-" + args.id + "-" + timestr + "-" + "/"
model_directory = os.path.dirname(os.path.realpath(__file__)) + "/" + datestr + "-dncf-acer-models-" + args.id + "-" + timestr + "-" + "/"

TIMESTEPS = args.timesteps


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    # default n steps of ACER: 20
    global best_mean_reward, n_steps
    if (n_steps + 1) % 100 == 0:
        x, y = ts2xy(load_results(log_directory), 'timesteps')
        if len(x) > 0:
            mean_reward = numpy.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
                best_mean_reward, mean_reward))

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print("Saving new best model")
            model.save(model_directory + 'acer-model_' + str((n_steps + 1) * 20))  # default n steps of ACER: 20
    n_steps += 1
    return True


if __name__ == "__main__":
    os.makedirs(log_directory, exist_ok=True)
    os.makedirs(model_directory, exist_ok=True)

    env = gym.make('ModuleSelect-v1',
                    verbose=0,
                    )
    env = Monitor(env, log_directory, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    model = ACER(
        env=env,
        policy=MlpPolicy,
        verbose=1,
        tensorboard_log="./" + args.id + "_dcnf_acer_tensorboard/",
        learning_rate=args.learning_rate,
        buffer_size=10000,  # test
    )
    print("Learning Rate:", args.learning_rate)
    try:
        model.learn(
            total_timesteps=TIMESTEPS,
            callback=callback
        )
    except KeyboardInterrupt:
        pass
    finally:
        env.envs[0].env.close()
