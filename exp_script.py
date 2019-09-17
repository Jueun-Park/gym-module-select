import argparse
import os
import csv
import time


def init_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max-num-proc', help='max num of processes', type=int, default=10)
    args = parser.parse_args()
    return args

args = init_parse_argument()

for num_proc in range(args.max_num_proc + 1):
    os.system("python play_const_proc_each_action.py -e 10 -p " + str(num_proc))
