from gym.envs.registration import register

try:
    register(
        id='DonkeyVae-v0',
        entry_point='gym_vae_donkey.envs.vae_env.vae_env:DonkeyVAEEnv',
        max_episode_steps=None,
        )
except:
    pass
