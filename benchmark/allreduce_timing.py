import argparse
import os
from time import time_ns

import horovod.torch as hvd
import numpy as np
import torch


def non_empty_string(s):
    if not s:
        raise ValueError("Must not be empty string")
    return s


parser = argparse.ArgumentParser(description='Benchmark horovod allreduce operation.')
parser.add_argument('system', help="The name of the current system", type=non_empty_string)
args = parser.parse_args()

if not os.path.exists(f"data/{args.system}"):
    os.mkdir(f"data/{args.system}")

hvd.init()

rank = hvd.rank()
size = hvd.size()

scales = np.logspace(2, 4, 20, dtype=int)
num_rep = 10

results = np.zeros((len(scales), num_rep))
for j, scale in enumerate(scales):
    a = torch.randn(scale, scale)

    # warmup
    for i in range(5):
        a = hvd.allreduce(a)

    for i in range(num_rep):
        if rank == 0:
            start = time_ns()

        a = hvd.allreduce(a)

        if rank == 0:
            end = time_ns()
            time = (end - start) * 1E-9
            results[j][i] = time

if rank == 0:
    mean_time = results.mean(axis=1)
    std_time = results.std(axis=1)
    print("mean:", mean_time)
    print("std:", std_time)
    np.save(f"data/{args.system}/ar-time-result{size}.npy", [mean_time, std_time])
