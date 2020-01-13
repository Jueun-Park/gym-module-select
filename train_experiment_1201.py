import os

# 4.5h * 6 = 27h
timesteps = 320001
lrs = [8e-5, 9e-5, 1e-4, 2e-4]
for lr in lrs:
    os.system("python train/train_dqn.py -l " + str(lr) + " -t " + str(timesteps) + " -i 01exp")
