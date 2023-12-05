import torch
import torchvision
import itertools

from augmentations import HumbleTransforms



def group_dataitem_by_class(data_list):

    sorted_tuples = sorted(data_list, key=lambda x: x[1])
    n = max(sorted_tuples, key=lambda x: x[1])[1]+1

    grouped_tuples = [list(group) for _, group in itertools.groupby(sorted_tuples, key=lambda x: x[1])]

    assert n == len(grouped_tuples), f"unmatched N {n}; G {len(grouped_tuples)}"
    return grouped_tuples



class MNISTkind_JFedDatset(torch.utils.data.Dataset):
    """For datasets form torch data"""
    def __init__(self, data_path: str = None,
                    dataset_type = "MNIST",
                    center = "all",
                    total_centers = 5,
                    iid_ness = "full", # full / semi / non
                    semi_client_per_class = 2, # only for semi iid_ness
                    split_type: str = "cls_train",
                    seed = 59,
                    transform=None):

        self.center = center
        self.total_centers = total_centers
        self.transform = transform
        self.train = True if split_type == "cls_train" else False

        if dataset_type == "MNIST":
            self.full_dataset = torchvision.datasets.MNIST(root=data_path,
                                    train=self.train, download=True, transform=transform)
        elif dataset_type == "FashionMNIST":
            self.full_dataset = torchvision.datasets.FashionMNIST(root=data_path,
                                                train=self.train, download=True, transform=transform)
        elif dataset_type == "EMNIST":
            self.full_dataset = torchvision.datasets.EMNIST(root=data_path, split="balanced",
                                                train=self.train, download=True, transform=transform)


        if (not self.train) or (center=="all"):
            self.client_dataset = self.full_dataset
            return

        assert center < total_centers, f"center {center} doesnot exist for totalcenters {total_centers}"
        grouped_data = group_dataitem_by_class(self.full_dataset)
        cls_count = len(grouped_data)
        # print(dataset_type, ">>> Class Count:::", cls_count)
        assert cls_count >= total_centers, (f"Total Class {cls_count} < Total Centers {total_centers}; "
                                            "This will result in unexpected behaviour in non/semi iid-ness modes")

        self.client_dataset = []
        if iid_ness == "non":
            # if total center > classes then will return empty partitions
            # for all centers above the class count
            for i, gd in enumerate(grouped_data):
                if (i % total_centers) == center:
                    self.client_dataset.extend(gd)

        elif iid_ness == "full":
            for i, gd in enumerate(grouped_data):
                self.client_dataset.extend(gd[center::total_centers])

        elif iid_ness == "semi":
            # if total center > classes then will return non overlapping classes
            # thus outcome will result in non-iid type data partition
            data_chunk = []
            divsr = semi_client_per_class
            for i, gd in enumerate(grouped_data):
                chunk = len(gd)//divsr
                for j in range(divsr):
                    data_chunk.append(gd[j*chunk: (j+1)*chunk])
            flattened_list = [inner
                              for outer in data_chunk[center::total_centers]
                              for inner in outer]
            self.client_dataset.extend(flattened_list)

        else:
            raise f"unknown iidness specified {iid_ness}"


    def __len__(self):
        return len(self.client_dataset)

    def __getitem__(self, idx):
        return self.client_dataset[idx]



def dataset_partition_sanity_validator():
    center_count = 13
    counter=0
    imgset = set()
    fullclsset = set()
    for center in range(center_count):
        dataset = MNISTkind_JFedDatset(data_path="/home/joseph.benjamin/WERK/fed-cvpr/data/torch-data/",
                                    dataset_type='EMNIST',
                                    center=center,
                                    total_centers=center_count,
                                    iid_ness="semi",
                                    semi_client_per_class = 2,
                                    )
        len(dataset)

        cls = set()
        for i in range(len(dataset)):
            cls.add(dataset.__getitem__(i)[1])
            imgset.add(hash(dataset.__getitem__(i)[0].tobytes()))
            counter+=1
        # assert counter == len(imgset), f"{counter} yyy {len(imgset)}"
        print(f"{cls} Counter: {counter} UniqueImg: {len(imgset)}")
        fullclsset = fullclsset.union(cls)
    print(fullclsset)

