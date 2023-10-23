import os
import copy
import torch


def global_average_weights(w:list, device = "cpu"):
    """
    w: list of pytorch parameters for weighs
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key].to(device)
        for i in range(1, len(w)):
            w_avg[key] += w[i][key].to(device)
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def gpu_devices_generator():
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_ids = 0 if gpu_ids == None else gpu_ids.split(",")
    gpu_devices = [torch.device(f"cuda:{gid}") for gid in range(len(gpu_ids))]
    print("Enabled CUDA Devices::", gpu_ids)
    index = 0
    while True:
        yield gpu_devices[index]
        index = (index + 1) % len(gpu_devices)