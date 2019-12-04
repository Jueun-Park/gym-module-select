import os

# about: 2h * 10 = 20h (cpa=1)
timesteps = 1000001
lr = 1e-4
for i in range(10):
    os.system("python train/train_dqn.py -l " + str(lr) + " -t " + str(timesteps))
