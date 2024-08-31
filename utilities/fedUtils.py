import os
import copy
import torch
import numpy as np

def gpu_devices_generator():
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_ids = 0 if gpu_ids == None else gpu_ids.split(",")
    gpu_devices = [torch.device(f"cuda:{gid}") for gid in range(len(gpu_ids))]
    print("Enabled CUDA Devices::", gpu_ids)
    index = 0
    while True:
        yield gpu_devices[index]
        index = (index + 1) % len(gpu_devices)

class GPUDeviceGenerator:
    def __init__(self):
        gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES')
        gpu_ids = "0" if gpu_ids is None else gpu_ids.split(",")
        self.gpu_devices = [torch.device(f"cuda:{gid}") for gid in range(len(gpu_ids))]
        self.index = 0
        print("Enabled CUDA Devices:", gpu_ids)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.gpu_devices:
            raise StopIteration("No more GPU devices available.")
        device = self.gpu_devices[self.index]
        self.index = (self.index + 1) % len(self.gpu_devices)
        return device

    def pop_device(self, device):
        if device in self.gpu_devices:
            self.gpu_devices.remove(device)
            self.index = self.index % len(self.gpu_devices) if self.gpu_devices else 0
            print(f"Device {device} popped. Remaining devices: {self.gpu_devices}")
        else:
            print(f"Device {device} not found in the list.")



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


def create_dirichlet_federated_split( data_name_list, label_list,
                            dirichlet_parameter: float = 1,
                            num_clients: int = 10):
    """ Adapted from https://github.com/google-research/federated/blob/master/utils/datasets/cifar10_dataset.py
    """

    uniq_classes = set(label_list)
    num_classes = len(uniq_classes)
    num_samples = len(data_name_list)
    dataname_array = np.asarray(data_name_list)
    labels_array = np.asarray(label_list) #ease of use

    train_clients = {}
    train_multinomial_vals = []
    # Each client has a multinomial distribution over classes drawn from a
    # Dirichlet.
    for k in range(num_clients):
        proportion = np.random.dirichlet(dirichlet_parameter *
                                        np.ones(num_classes,))
        train_multinomial_vals.append(proportion)
        print(proportion)

    train_multinomial_vals = np.array(train_multinomial_vals)

    train_example_indices = []
    xamples_at_label_c = {}
    for c in uniq_classes:
        train_label_c = np.where(labels_array == c)[0]
        np.random.shuffle(train_label_c)
        train_example_indices.append(train_label_c)
        xamples_at_label_c[c] = label_list.count(c)

    train_example_indices = np.asarray(train_example_indices)
    train_client_samples = [[] for _ in range(num_clients)]
    train_count = np.zeros(num_classes).astype(int)
    train_examples_per_client = int(num_samples / num_clients)

    for k in range(num_clients):
        for i in range(train_examples_per_client):
            sampled_label = np.argwhere(
                    np.random.multinomial(1, train_multinomial_vals[k, :]) == 1)[0][0]
            ### Safetynet
            # while_counter =0
            # while train_count[sampled_label] >  xamples_at_label_c[sampled_label]:
            #     print(np.random.multinomial(1, train_multinomial_vals[k, :]))
            #     sampled_label = np.argwhere(
            #         np.random.multinomial(1, train_multinomial_vals[k, :]) == 1)[0][0]
            #     while_counter+=1
            #     if while_counter>1000: break
            ##---
            train_client_samples[k].append(
                train_example_indices[sampled_label][train_count[sampled_label]])
            train_count[sampled_label] += 1
            if train_count[sampled_label] == xamples_at_label_c[sampled_label]:
                train_multinomial_vals[:, sampled_label] = 0
                train_multinomial_vals = (
                    train_multinomial_vals /
                    train_multinomial_vals.sum(axis=1)[:, None])

    for k in range(num_clients):
        client_name = k
        x_train = dataname_array[np.array(train_client_samples[k])]
        y_train = labels_array[np.array(
            train_client_samples[k])].astype('int64').squeeze()
        train_data = {'image': x_train.tolist(), 'label': y_train.tolist()}
        train_clients[client_name] = train_data

    print(train_clients)
    return train_clients