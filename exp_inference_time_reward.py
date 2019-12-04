import os


NUM_EXPERIMENT_EPISODES = 10
# times = [0.04, 0.045, 0.05, 0.055, 0.06]
times = [0.065, 0.07, 0.075, 0.08]
for delay in times:
    os.system("python play_simple_agent.py -e " + str(NUM_EXPERIMENT_EPISODES) + " -m 0 -d " + str(delay))
