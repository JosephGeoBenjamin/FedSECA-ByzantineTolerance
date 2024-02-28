import os, sys

import torch
import torchvision
import itertools
import numpy as np

sys.path.append(os.getcwd())
import utilities.logUtils as lutl

from datacode.augmentations import HumbleAuguments

##==============================================================================
#NOTE: No seperate validation codes


class MNISTkind_JFedDatset(torch.utils.data.Dataset):
    """For datasets form torch data"""
    def __init__(self, data_path: str = None,
                    dataset_type = "MNIST",
                    center = "all",
                    total_centers = 5,
                    iid_ness = "full", # full / semi-mix / semi-pure / non
                    semi_client_per_class = 2, # only for semi iid_ness
                    split_type: str = "cls_train",
                    seed = 59,
                    transforms=None):

        self.center = center
        self.total_centers = total_centers
        self.iid_ness = iid_ness
        self.semi_client_per_class = semi_client_per_class
        self.transforms = transforms

        if   split_type == "cls_train": self.train = True
        elif split_type == "cls_test" : self.train = False
        else: raise Exception(f"Unknown Split type specified, {split_type}")

        if dataset_type == "MNIST":
            self.full_dataset = torchvision.datasets.MNIST(root=data_path,
                                    train=self.train, download=True, transform=transforms)
        elif dataset_type == "FashionMNIST":
            self.full_dataset = torchvision.datasets.FashionMNIST(root=data_path,
                                                train=self.train, download=True, transform=transforms)
        elif dataset_type == "EMNIST":
            self.full_dataset = torchvision.datasets.EMNIST(root=data_path, split="balanced",
                                                train=self.train, download=True, transform=transforms)

        if (not self.train) or (center=="all"):
            self.client_dataset = self.full_dataset
        else:
            assert center < total_centers, f"center {center} doesnot exist for totalcenters {total_centers}"

            self.client_dataset = self._split_based_on_iidness()


        self.targets = [d[1] for d in self.client_dataset]



    def group_dataitem_by_class(self, data_list):

        sorted_tuples = sorted(data_list, key=lambda x: x[1])
        n = max(sorted_tuples, key=lambda x: x[1])[1]+1

        grouped_tuples = [list(group) for _, group in itertools.groupby(sorted_tuples, key=lambda x: x[1])]

        assert n == len(grouped_tuples), f"unmatched N {n}; G {len(grouped_tuples)}"
        return grouped_tuples


    def _split_based_on_iidness(self):
        client_dataset = []
        grouped_data = self.group_dataitem_by_class(self.full_dataset)
        cls_count = len(grouped_data)
        assert cls_count >= self.total_centers, (f"Total Class {cls_count} < Total Centers {self.total_centers}; "
                                            "This will result in unexpected behaviour in non/semi iid-ness modes")


        if self.iid_ness == "non":
            # if total center > classes then will return empty partitions
            # for all centers above the class count
            for i, gd in enumerate(grouped_data):
                if (i % self.total_centers) == self.center:
                    client_dataset.extend(gd)


        elif self.iid_ness == "full":
            for i, gd in enumerate(grouped_data):
                client_dataset.extend(gd[self.center::self.total_centers])


        elif self.iid_ness == "semi-mix":
            # if total center > classes then will return non overlapping classes
            # thus outcome will result in non-iid type data partition
            data_chunk = []
            divsr = self.semi_client_per_class
            for i, gd in enumerate(grouped_data):
                chunk = len(gd)//divsr
                for j in range(divsr):
                    data_chunk.append(gd[j*chunk: (j+1)*chunk])
            flattened_list = [inner
                              for outer in data_chunk[self.center::self.total_centers]
                              for inner in outer]
            client_dataset.extend(flattened_list)


        elif self.iid_ness == "semi-pure":
            # will try to approximate the data counts in client but not guarenteed always
            divsr = self.semi_client_per_class
            class_chunk_est = [[] for i in range(self.total_centers)]
            chunk_chi = []
            group_ids = list(range(len(grouped_data)))
            center_pointer = 0

            cls_size = int(np.ceil(len(group_ids) / int(np.ceil(self.total_centers/divsr)) ))
            cls_groups = [group_ids[i:i + cls_size] for i in range(0, len(group_ids), cls_size)]

            while center_pointer<self.total_centers:
                grp = cls_groups.pop(0)
                for j in range(divsr):
                    if center_pointer< self.total_centers:
                        class_chunk_est[center_pointer] = grp
                        chunk_chi.append(j)
                        center_pointer+=1
            # print(class_chunk_est)

            class_chunk_est = [ (clss, class_chunk_est.count(clss), chunk_chi[i])
                            for i, clss in enumerate(class_chunk_est)]

            clses, chunks, chi = class_chunk_est[self.center]
            flattened_list = []
            for cls in clses:
                gd = grouped_data[cls]
                chunk_sz = len(gd) // chunks
                flattened_list.extend(gd[chi*chunk_sz:(chi+1)*chunk_sz])

            client_dataset.extend(flattened_list)


        else:
            raise f"unknown iidness specified {self.iid_ness}"

        return client_dataset




    def __len__(self):
        return len(self.client_dataset)

    def __getitem__(self, idx):
        return self.client_dataset[idx]


##------------------------------------------------------------------------------

def getHumbleCLSLoaders(cfg, center_index = None, override_csv = None):
    """ center_index: None/all --> pooled
    """
    if center_index == None: center_index = "all"

    data_path     = cfg.data_root_path
    info_log_path = cfg.gLogPath +'/misc_data.txt'
    img_size_in   = cfg.image_size
    batch_size    = cfg.batch_size
    workers       = cfg.workers
    total_centers = cfg.data_centers_count
    num_classes   = cfg.clsfy_layers[-1] #10
    override_csv  = cfg.override_csv if cfg.override_csv else None

    assert (override_csv == None), "Override CSV method Not implemented"

    traindataset = MNISTkind_JFedDatset(data_path  = data_path,
                                dataset_type  = cfg.dataset_type,
                                split_type    = "cls_train",
                                center        = center_index,
                                total_centers = total_centers,
                                iid_ness      = cfg.iid_ness,
                                semi_client_per_class = 2,
                                transforms=HumbleAuguments()
                                )


    validdataset = MNISTkind_JFedDatset(data_path  = data_path,
                                dataset_type  = cfg.dataset_type,
                                split_type    = "cls_test",
                                center        = center_index,
                                transforms=HumbleAuguments()
                                )

    # class_weights = get_class_weights(traindataset.targets, nclasses=num_classes)

    trainloader  = torch.utils.data.DataLoader( traindataset, shuffle=True,
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=True, persistent_workers=False)

    validloader  = torch.utils.data.DataLoader( validdataset, shuffle=False,
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=True)

    lutl.LOG2DICTXT({"DC":(cfg.data_type, center_index), "Train-":len(traindataset),
                    "TargetClasses": str(set(traindataset.targets)),
                    "Transform": str(traindataset.transforms.get_composition()),
                    #  "class-weights":str(class_weights)
                     }, info_log_path)
    lutl.LOG2DICTXT({"DC":(cfg.data_type, center_index), "Valid-":len(validdataset),
                    "TargetClasses": str(set(traindataset.targets)),
                    "Transform": str(validdataset.transforms.get_composition()),
                     }, info_log_path)

    if override_csv:
        lutl.LOG2TXT(f"OverRide CSV-set: {override_csv} !^!^!^!", info_log_path)

    return trainloader, validloader



def getHumbleTESTLoader(cfg, center_index = None):
    """ center_index: None/all --> pooled
    """
    if center_index == None: center_index = "all"

    data_path = cfg.data_root_path
    info_log_path = cfg.gLogPath +'/misc_data.txt'
    img_size_in   = cfg.image_size
    batch_size = cfg.batch_size
    workers    = cfg.workers

    dataset = MNISTkind_JFedDatset(data_path  = data_path,
                                dataset_type  = cfg.dataset_type,
                                split_type    = "cls_test",
                                center        = center_index,
                                transforms     = HumbleAuguments()
                                )

    testloader  = torch.utils.data.DataLoader(dataset, shuffle=False,
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=True)

    lutl.LOG2DICTXT({"DC":(cfg.data_type, center_index), "TEST-":len(dataset),
                    "TargetClasses": str(set(dataset.targets)),
                    "Transform": str(dataset.transforms.get_composition()),
                     }, info_log_path)

    return testloader



##==============================================================================



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

