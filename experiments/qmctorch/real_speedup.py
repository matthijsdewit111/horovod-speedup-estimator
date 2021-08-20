import os
from time import time_ns

import horovod.torch as hvd
import numpy as np
import torch

from slater_jastrow_wrapper import SlaterJastrowWrapperH2, SlaterJastrowWrapperCH4


def dummy_training(model: torch.nn.Module, optimizer, batch_size, input_size):
    input_data = torch.randn(batch_size, input_size)

    training_cycle_time_results = []
    for _ in range(100):
        start = time_ns()
        model.zero_grad()
        output = model(input_data)
        loss = torch.functional.F.mse_loss(output, output)
        loss.backward()
        optimizer.step()
        end = time_ns()
        training_cycle_time_results.append((end - start) * 1E-9)

    return np.median(training_cycle_time_results)


def run_sequential():
    sjh2 = SlaterJastrowWrapperH2()
    optimizer = torch.optim.SGD(sjh2.parameters(), lr=0.05)

    h2_results = []
    for bs in batch_size_range:
        time = dummy_training(sjh2, optimizer, bs, 6)
        h2_results.append(time)

    sjch4 = SlaterJastrowWrapperCH4()
    optimizer = torch.optim.SGD(sjch4.parameters(), lr=0.05)

    ch4_results = []
    for bs in batch_size_range:
        time = dummy_training(sjch4, optimizer, bs, 30)
        ch4_results.append(time)

    return h2_results, ch4_results


def run_distributed():
    sjh2 = SlaterJastrowWrapperH2()
    optimizer = torch.optim.SGD(sjh2.parameters(), lr=0.05)

    hvd.broadcast_parameters(sjh2.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd_optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=sjh2.named_parameters())

    h2_results = []
    for bs in batch_size_range:
        time = dummy_training(sjh2, hvd_optimizer, bs, 6)
        h2_results.append(time)

    sjch4 = SlaterJastrowWrapperCH4()
    optimizer = torch.optim.SGD(sjch4.parameters(), lr=0.05)

    hvd.broadcast_parameters(sjch4.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd_optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=sjch4.named_parameters())

    ch4_results = []
    for bs in batch_size_range:
        time = dummy_training(sjch4, hvd_optimizer, bs, 30)
        ch4_results.append(time)

    return h2_results, ch4_results


if __name__ == "__main__":
    hvd.init()
    size = hvd.size()
    rank = hvd.rank()

    batch_size_range = np.linspace(1, 400, num=50, dtype=int)

    if size > 1:
        h2r, ch4r = run_distributed()
    else:
        h2r, ch4r = run_sequential()

    if not os.path.exists(f"real_data/"):
        os.mkdir("real_data")

    np.save(f"real_data/h2_time_result_{size}.npy", h2r)
    np.save(f"real_data/ch4_time_result_{size}.npy", ch4r)
