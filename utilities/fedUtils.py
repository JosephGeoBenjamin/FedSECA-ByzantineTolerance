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



def find_layerwise_weight_difference(m1, m2):
    """ L2 Norm
    """
    m1state = m1.state_dict()
    m2state = m2.state_dict()
    layerwise = {}
    for k in m1state:
        try:
            er = torch.norm(m1state[k] - m2state[k], p=2)
        except:
            er = torch.tensor(-1.0)
        layerwise[k] = er.item()

    return layerwise