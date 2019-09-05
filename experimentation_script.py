import os

commands = {0: "python lane_tracker_single_module.py",
            1: "python vae_sac_single_module.py",
            }

for num_command in range(2):
    for delay in range(0, 101, 10):
        os.system(commands[num_command] + " -d " + str(delay))
