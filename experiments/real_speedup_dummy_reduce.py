import os
import sys
from time import time_ns

import horovod.torch as hvd
import numpy as np
import torch

sys.path.append("../examples")
from some_module import MyLargeNet
from some_module import MySmallNet


def dummy_training(model: torch.nn.Module, optimizer, batch_size):
    input_data = torch.randn(batch_size, 100)

    training_cycle_time_results = []
    for _ in range(100):
        start = time_ns()
        model.zero_grad()
        output = model(input_data)
        loss = torch.functional.F.mse_loss(output, output)
        loss.backward()
        end = time_ns()
        training_cycle_time_results.append((end - start) * 1E-9)
        optimizer.step()  # error if no step

    return np.median(training_cycle_time_results)


def run_sequential():
    smallnet = MySmallNet()
    optimizer = torch.optim.SGD(smallnet.parameters(), lr=0.05)

    smallnet_results = []
    for bs in batch_size_range:
        time = dummy_training(smallnet, optimizer, bs)
        smallnet_results.append(time)

    largenet = MyLargeNet()
    optimizer = torch.optim.SGD(largenet.parameters(), lr=0.05)

    largenet_results = []
    for bs in batch_size_range:
        time = dummy_training(largenet, optimizer, bs)
        largenet_results.append(time)

    return smallnet_results, largenet_results


def run_distributed():
    smallnet = MySmallNet()
    optimizer = torch.optim.SGD(smallnet.parameters(), lr=0.05)

    hvd.broadcast_parameters(smallnet.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd_optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=smallnet.named_parameters())

    smallnet_results = []
    for bs in batch_size_range:
        time = dummy_training(smallnet, hvd_optimizer, bs)
        smallnet_results.append(time)

    largenet = MyLargeNet()
    optimizer = torch.optim.SGD(largenet.parameters(), lr=0.05)

    hvd.broadcast_parameters(largenet.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd_optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=largenet.named_parameters())

    largenet_results = []
    for bs in batch_size_range:
        time = dummy_training(largenet, hvd_optimizer, bs)
        largenet_results.append(time)

    return smallnet_results, largenet_results


if __name__ == "__main__":
    hvd.init()
    size = hvd.size()
    rank = hvd.rank()

    batch_size_range = np.linspace(1, 400, num=50, dtype=int)

    if size > 1:
        snr, lnr = run_distributed()
    else:
        snr, lnr = run_sequential()

    if not os.path.exists(f"real_data/"):
        os.mkdir("real_data")

    np.save(f"real_data/small_net_time_result_{size}_dummy_reduce.npy", snr)
    np.save(f"real_data/large_net_time_result_{size}_dummy_reduce.npy", lnr)
