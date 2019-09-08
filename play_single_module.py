import numpy as np
import gym
from modules.vae_sac_modules import VAESACModule
from utils.utils import create_test_env, get_saved_hyperparams, ALGOS

# TODO: module_select_env.py 를 통해 단일 모듈을 실행시키는 스크립트


if __name__ == "__main__":
    # test code
    stats_path = "modules/logs/sac/DonkeyVae-v0-level-0_6/DonkeyVae-v0-level-0"
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False)
    hyperparams['vae_path'] = "modules/logs/vae-level-0-dim-32.pkl"
    inner_env = create_test_env(stats_path=stats_path,
                                seed=0,
                                log_dir="modules/logs",
                                hyperparams=hyperparams)
    model_path = "modules/logs/sac/DonkeyVae-v0-level-0_6/DonkeyVae-v0-level-0.pkl"
    model = ALGOS["sac"].load(model_path)

    module1 = VAESACModule(inner_env, model, 5)

    inner_obs = inner_env.reset()
    for i in range(1000):
        inner_action = module1.predict(inner_obs)
        if isinstance(inner_env.action_space, gym.spaces.Box):
            inner_action = np.clip(inner_action, inner_env.action_space.low, inner_env.action_space.high)
        inner_obs, reward, done, infos = inner_env.step(inner_action)
