from gym.envs.registration import register

register(
    id='DonkeyVae-v0',
    entry_point='gym_module_select.envs.vae_env.vae_env:DonkeyVAEEnv',
    max_episode_steps=None,
)

register(
    id='ModuleSelect-v0',
    entry_point='gym_module_select.envs.module_select_env:ModuleSelectEnv',
)

register(
    id='ModuleSelectContinuous-v0',
    entry_point='gym_module_select.envs.module_select_env:ModuleSelectEnvContinuous',
)
