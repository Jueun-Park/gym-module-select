import os


# 0 to 4, total 5 models
for num_command in range(6):
    os.system("python play_simple_agent.py -m " + str(num_command))
