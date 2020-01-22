from os import system


# about 24h for one experiment, total 3 days (?!)
cpa_and_timestep = [
    [5, 400001],
    [10, 200001],
    [20, 100001],
]

for cpa, timestep in cpa_and_timestep:
    system("python train/train_acer.py -i test_cpa --cpa " + str(cpa) + " -t " + str(timestep))
