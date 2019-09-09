import os


for num_command in range(5):
    os.system("python play_simple_agent.py -m " + str(num_command))
