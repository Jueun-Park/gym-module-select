import numpy as np
import time

def add_delay(proc_state, delay_weight):
    proc_state /= 100
    wait_time = np.random.exponential(scale=proc_state)
    wait_time *= delay_weight
    time.sleep(wait_time)