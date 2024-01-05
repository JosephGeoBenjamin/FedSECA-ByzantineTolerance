import sys, os
import subprocess

import random
import numpy as np
import torch


def START_SEED(seed=73):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_train_param(model, console = False):
    train_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if console:
        print('The model has {} trainable parameters'.format(train_params_count))
    return train_params_count



##==============================================================================


class ObjDict(dict):
    """
    reference: https://stackoverflow.com/a/32107024
    """
    def __init__(self, *args, **kwargs):
        super(ObjDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(ObjDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(ObjDict, self).__delitem__(key)
        del self.__dict__[key]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    # TODO: commenting for Safety Add back later
    # def update(self, *args, **kwargs):
    #     for k, v in dict(*args, **kwargs).items():
    #         self.__setitem__(k, v)


##======================= NVIDIA - GPU =========================================

# Check if the visible devices are free
def get_devices_freespace(free_need=40000):
    cmd = "nvidia-smi --query-gpu=memory.free --format=csv"
    # cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
    output = subprocess.check_output(cmd.split()).decode("utf-8").strip()
    lines = output.split("\n")[1:]  # Skip header line
    memory_free = [int(line.split()[0]) for line in lines]
    free_devices = [{"GPU-ID":i, "MemoryFree":mem}
                    for i, mem in enumerate(memory_free) if mem > free_need ]
    return free_devices

def get_devices_usagemin(usage_limit=500):
    cmd = "nvidia-smi --query-gpu=memory.used --format=csv"
    output = subprocess.check_output(cmd.split()).decode("utf-8").strip()
    lines = output.split("\n")[1:]  # Skip header line
    memory_used = [int(line.split()[0]) for line in lines]
    free_devices = [{"GPU-ID":i, "MemoryUsed":mem}
                    for i, mem in enumerate(memory_used) if mem < usage_limit ]
    return free_devices


