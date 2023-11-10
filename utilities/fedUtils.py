import os
import copy
import torch


def gpu_devices_generator():
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_ids = 0 if gpu_ids == None else gpu_ids.split(",")
    gpu_devices = [torch.device(f"cuda:{gid}") for gid in range(len(gpu_ids))]
    print("Enabled CUDA Devices::", gpu_ids)
    index = 0
    while True:
        yield gpu_devices[index]
        index = (index + 1) % len(gpu_devices)