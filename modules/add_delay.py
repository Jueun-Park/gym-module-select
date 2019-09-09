import numpy as np
import time


def add_delay(num_proc, delay_weight, static_term):
    num_proc /= 100
    wait_time = np.random.exponential(scale=num_proc)
    wait_time *= delay_weight
    wait_time += static_term
    time.sleep(wait_time)
