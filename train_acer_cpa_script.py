from os import system


# about 24h for one experiment, total 3 days (?!)
# cpa_and_timestep = [
#     [5, 400001],
#     [10, 200001],
#     [20, 100001],
# ]

# for cpa, timestep in cpa_and_timestep:
#     system("python train/train_acer.py -i test_cpa --cpa " + str(cpa) + " -t " + str(timestep))

cpa = 5
timestep = 300001  # 300k, about 21h for one expr
# timestep = 10
learning_rates = [1e-4, 2e-4, 4e-4]
for lr in learning_rates:
    system("python train/train_acer.py -i lr_exp -l " + str(lr) + " --cpa " + str(cpa) + " -t " + str(timestep))
