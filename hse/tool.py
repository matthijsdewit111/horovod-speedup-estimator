import gc
import importlib
from time import time_ns

import numpy as np
import torch
from scipy.optimize import curve_fit

from hse.argument_parsing import get_args
from hse.fit_functions import stepped_linear_2d
from hse.plotting import plot_prediction, plot_predicted_speedup

ignore_2_and_3 = True


def get_model():
    module = importlib.import_module(args.module)
    model = getattr(module, args.model)()

    if not isinstance(model, torch.nn.Module):
        raise ValueError("model is not a subclass of torch.nn.Module")
    return model


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_real_data(system):
    num_procs = list(range(1, 33))

    means = []
    for num_proc in num_procs:
        if ignore_2_and_3 and num_proc in [2, 3]:
            continue
        results_mean, _ = np.load(f"../benchmark/data/{system}/ar-time-result{num_proc}.npy")
        means.append(results_mean)

    return np.array(means).T


def estimate_parameters(X, Y, means):
    X_mesh, Y_mesh = np.meshgrid(X, Y)
    xdata = np.stack((X_mesh, Y_mesh), axis=2).reshape(-1, 2)
    popt, _ = curve_fit(stepped_linear_2d, xdata, means)

    if args.verbose:
        plot_prediction(stepped_linear_2d, xdata, popt, X_mesh, Y_mesh, means.reshape(original_shape))
    return popt


def estimate_computation_time(model, batch_size):
    if args.input_size:
        input_size = args.input_size
    else:
        input_layer = model.__getattr__(args.input_layer)
        input_size = (input_layer.in_features,)

    input_shape = (batch_size,) + tuple(input_size)
    input_data = torch.randn(input_shape)

    loss_func = getattr(torch.functional.F, args.loss_function)

    gc.collect()

    training_cycle_time_results = []
    for _ in range(args.iterations):
        start = time_ns()
        output = model(input_data)
        loss = loss_func(output, output)
        loss.backward()
        end = time_ns()
        training_cycle_time_results.append((end - start) * 1E-9)

    return np.median(training_cycle_time_results)


if __name__ == "__main__":
    args = get_args()

    model = get_model()

    mean_communication_times = get_real_data(args.system)
    original_shape = mean_communication_times.shape
    mean_communication_times = mean_communication_times.flatten()

    processes_tested = np.append(np.array([1]), np.arange(4, 33)) if ignore_2_and_3 else np.arange(1, 33)
    message_sizes_tested = np.logspace(2, 4, 20, dtype=int) ** 2
    estimated_params = estimate_parameters(processes_tested, message_sizes_tested, mean_communication_times)

    num_param = count_trainable_parameters(model)
    processes_range = np.linspace(2, args.max_processes, num=min(args.max_processes - 1, 50), dtype=int)
    num_model_parameters = np.full((len(processes_range),), num_param)
    data_to_predict = np.stack((processes_range, num_model_parameters), axis=1)
    predicted_communication_times = stepped_linear_2d(data_to_predict, *estimated_params)

    batch_size_range = np.linspace(1, args.max_batch_size, num=min(args.max_batch_size, 50), dtype=int)
    computation_times_per_batch = np.array([estimate_computation_time(model, bs) for bs in batch_size_range])

    speedup = np.zeros((len(processes_range), len(batch_size_range)))
    for i, p in enumerate(processes_range):
        for j in range(len(batch_size_range)):
            s = computation_times_per_batch[j] / ((predicted_communication_times[i] + computation_times_per_batch[j]) / p)
            speedup[i][j] = s if s >= 1 else np.nan

    plot_predicted_speedup(speedup, batch_size_range, processes_range, save_as=args.save_as, cmap_name=args.color_map)
