import gym
import gym_module_select

from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('ModuleSelect-v0')

model = DQN(
    env=env,
    policy=MlpPolicy,
    verbose=1,
)

model.learn(total_timesteps=1000)

env.close()
print("save the model")
model.save("test_dqn_model.pkl")

del model
model = DQN.load("test_dqn_model.pkl")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()
