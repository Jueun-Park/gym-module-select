from os import system


NUM_EXP = 50

acer_agent_700k = "train/20200103-dncf-acer-models-buf10k-194212-/acer-model_700000.zip"
acer_agent_1_4M = "train/20200103-dncf-acer-models-buf10k-194212-/acer-model_1410000.zip"

envs = ["dc", "df", "nc", "nf"]
for env in envs:
    input(">> waiting next " + env + " env...")
    system("python train/load_acer.py -m " + acer_agent_700k + " -e " + str(NUM_EXP))
    system("python train/load_acer.py -m " + acer_agent_1_4M + " -e " + str(NUM_EXP))

