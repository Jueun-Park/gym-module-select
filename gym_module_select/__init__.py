from gym.envs.registration import register


register(
    id='ModuleSelect-v1',
    entry_point='gym_module_select.envs.module_select_env:ModuleSelectEnv',
)

register(
    id='ModuleSelectContinuous-v1',
    entry_point='gym_module_select.envs.module_select_env:ModuleSelectEnvContinuous',
)
