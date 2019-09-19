import os


NUM_EXPERIMENT_EPISODES = 25

os.system("python play_heuristic_agent.py -e " + str(NUM_EXPERIMENT_EPISODES))
os.system("python play_greedy_agent.py -e " + str(NUM_EXPERIMENT_EPISODES))

# 0 to 4 and random agent, total 6 models
for num_command in range(6):
    os.system("python play_simple_agent.py -e " + str(NUM_EXPERIMENT_EPISODES) + " -m " + str(num_command))

# os.system("python train/load_sac.py -e " + str(NUM_EXPERIMENT_EPISODES) + " -m train/sac-models-164441-20190916/sac-model_52000.pkl")
# os.system("python train/load_sac.py -e " + str(NUM_EXPERIMENT_EPISODES) + " -m train/sac-models-164441-20190916/sac-model_88500.pkl")
