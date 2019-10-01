import os


NUM_EXPERIMENT_EPISODES = 25

# 0 to 1 and random agent, total 3 agents
for num_command in range(3):
    os.system("python play_simple_agent.py -e " + str(NUM_EXPERIMENT_EPISODES) + " -m " + str(num_command))
