import os

# 4.5h * 6 = 27h
timesteps = 320001
lr = 1e-4
for i in range(6):
    os.system("python train/train_dqn.py -l " + str(lr) + " -t " + str(timesteps))
