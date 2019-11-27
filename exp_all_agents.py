import os


NUM_EXPERIMENT_EPISODES = 25

# 0 to 3 and random agent(4), total 5 agents
for num_command in range(5):
    os.system("python play_simple_agent.py -e " + str(NUM_EXPERIMENT_EPISODES) + " -m " + str(num_command))
