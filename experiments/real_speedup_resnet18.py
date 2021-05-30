import os
from time import time_ns

import horovod.torch as hvd
import numpy as np
import torch
from torchvision.models import resnet18


def dummy_training(model: torch.nn.Module, optimizer, batch_size):
    input_data = torch.randn(batch_size, 3, 224, 224)

    training_cycle_time_results = []
    for _ in range(10):
        start = time_ns()
        model.zero_grad()
        output = model(input_data)
        loss = torch.functional.F.mse_loss(output, output)
        loss.backward()
        optimizer.step()
        end = time_ns()
        training_cycle_time_results.append((end - start) * 1E-9)

    return np.median(training_cycle_time_results)


def run_distributed():
    resnet = resnet18()
    optimizer = torch.optim.SGD(resnet.parameters(), lr=0.05)

    hvd.broadcast_parameters(resnet.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd_optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=resnet.named_parameters())

    resnet_results = []
    for bs in batch_size_range:
        time = dummy_training(resnet, hvd_optimizer, bs)
        resnet_results.append(time)

        # if rank == 0:
        #     r = resnet_results[-1]
        #     print(f"for batch size = {bs}:")
        #     print(f"Sec/Img: {r / bs}")
        #     print(f"Img/Sec/CPU: {bs / r}")
        #     print(f"Img/Sec/{size}CPU: {size * bs / r}")

    return resnet_results


if __name__ == "__main__":
    hvd.init()
    size = hvd.size()
    rank = hvd.rank()

    batch_size_range = np.linspace(1, 50, num=12, dtype=int)

    rnr = run_distributed()

    if rank == 0:
        if not os.path.exists(f"real_data/"):
            os.mkdir("real_data")

        np.save(f"real_data/resnet18_time_result_{size}.npy", rnr)
