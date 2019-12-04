import os


NUM_EXPERIMENT_EPISODES = 25
dqn_model_name = "train/ex-dn-dqn-models-005713-20191119/dqn-model_70000.pkl"
exp = ["True", "False"]
for ex in exp:
    os.system("python train/load_dqn.py -m " + dqn_model_name + " -e " + str(NUM_EXPERIMENT_EPISODES) + " -c " + ex)
