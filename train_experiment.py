import os

timesteps = 70001  # about: 10h 30m * 4 = 42h
# timesteps = 11
lr_list = [1e-4, 2e-4, 3e-4, 4e-4]
for lr in lr_list:
    os.system("python train/train_dqn.py -l " + str(lr) + " -t " + str(timesteps))
