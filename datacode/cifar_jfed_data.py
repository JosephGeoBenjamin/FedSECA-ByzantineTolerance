import time
import os, sys, pathlib
import random
import itertools
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter

import torch
import torchvision.transforms as torch_transforms
from torchvision.datasets.folder import default_loader as tv_image_loader
from torch.utils.data import ConcatDataset

sys.path.append(os.getcwd())
import utilities.logUtils as lutl
from datacode.augmentations import CifarClassifyAuguments
from utilities.metricUtils import get_class_weights


## ======================== Dataset Class ======================================

class Cifar_JFedDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path: str = None, #path to dataset images
        dataset_type: str = None, #cifar10 or cifar100
        csv_name = None, # for some special data partitioning
        center = "all",  # all--> Pooled
        split_type: str = "cls_train", # cls_train / cls_valid / test
        total_centers = 10,
        dirichlet_alpha = 100,
        iid_ness = "full",         # full / semi-mix / semi-pure / non
        semi_client_per_class = 2, # only for semi iid_ness
        label_type = "fine_label", # fine_label->100 / coarse_label->20 / label->10
        transforms=None,
    ):
        print(f"CIFAR Dataset Class for :: >> {dataset_type}")

        if dataset_type == "cifar10":
            assert (label_type == "label")
        if dataset_type == "cifar100":
            assert (label_type in ["fine_label", "coarse_label"])

        # cls_csv = csv_name if csv_name else "cifar100_clsfy_data.csv"
        ssl_csv  = csv_name if csv_name else f"{dataset_type}_ssl_data.csv"
        test_csv = csv_name if csv_name else f"{dataset_type}_test_data.csv"
        cls_csv  = csv_name if csv_name else f"{dataset_type}_train_data.csv"

        if data_path:
            if not (os.path.exists(data_path)):
                raise ValueError(f"The string {data_path} is not a valid path.")
            self.images_root = os.path.join(data_path, "train_images")
        else:
            raise ValueError(f"No path specified")

        if split_type == "ssl_train":
            df2 = pd.read_csv(os.path.join(data_path, ssl_csv))
        elif split_type == "cls_train":
            df2 = pd.read_csv(os.path.join(data_path, cls_csv))
            df2 = df2[df2["fold"] == "train"]
        elif split_type == "cls_valid":
            df2 = pd.read_csv(os.path.join(data_path, cls_csv))
            df2 = df2[df2["fold"] == "valid"]
        elif split_type == "test":
            self.images_root = os.path.join(data_path, "test_images") ##override
            df2 = pd.read_csv(os.path.join(data_path, test_csv))
            center = "all"
            print("Overriding set Center to ALL, since test is not grouped")
        else: raise("Unknown Split type specified ....", split_type)

        self.total_centers = total_centers
        self.center = center
        ## override semi_client_per_class
        self.semi_client_per_class = semi_client_per_class+2 if iid_ness=="semi-mix" else semi_client_per_class
        self.label_type = label_type
        self.pooled = True if center == "all" else False

        self.dirichlet_alpha = dirichlet_alpha
        self.iid_ness = iid_ness

        if not self.pooled:
            assert not (dirichlet_alpha and iid_ness), "Only one of `iid_ness` or `dirichlet` must be set"

            if (dirichlet_alpha is not None) and (dirichlet_alpha is not False):
                print("Using DIRICHLET based labelwise Split !!!")
                df2["center"] = df2[f"{dirichlet_alpha}_alpha_id"]
                df2["center"] = df2["center"].apply(self._remap_values,
                                                     max_value=df2["center"].max())
                df2 = df2[df2["center"] == center]

            elif (iid_ness is not None) and (iid_ness is not False):
                print("Using Manual IIDNESS labelwise Split !!!")
                df2 = self._split_based_on_iidness(df2)
                if not len(df2): print("Ignoring Empty data partition ..... &&&&&&")
            else:
                print("Defaulting to very random Split !!!")
                state = np.random.get_state(); np.random.seed(100)
                df2["center"] = np.random.randint(0, self.total_centers, size=len(df2))
                np.random.set_state(state)
                df2 = df2[df2["center"] == center]


        self.df2     = df2.reset_index(drop=True)
        self.targets = list(self.df2[label_type])
        self.transforms = transforms if transforms else torch_transforms.ToTensor()


    def group_dataitem_by_class(self, data_list):
        sorted_tuples = sorted(data_list, key=lambda x: x[-1])
        n = max(sorted_tuples, key=lambda x: x[-1])[-1]+1

        grouped_tuples = [list(group) for _, group in itertools.groupby(sorted_tuples, key=lambda x: x[-1])]

        assert n == len(grouped_tuples), f"unmatched N {n}; G {len(grouped_tuples)}"
        return grouped_tuples


    def _split_based_on_iidness(self, df):
        if len(df) == 0: return df # for validation ignoring

        client_dataset = []
        tuple_data =  df[['filename', 'image_id', self.label_type]].to_records(index=False)
        grouped_data = self.group_dataitem_by_class(tuple_data)


        cls_count = len(grouped_data)
        if self.iid_ness != "full":
            assert cls_count >= self.total_centers, (f"Total Class {cls_count} < Total Centers {self.total_centers}; "
                                            "This will result in unexpected behaviour in non/semi iid-ness modes")

        if self.iid_ness == "full":
            for i, gd in enumerate(grouped_data):
                client_dataset.extend(gd[self.center::self.total_centers])


        elif self.iid_ness == "non":
            # if total center > classes then will return empty partitions
            # for all centers above the class count
            for i, gd in enumerate(grouped_data):
                if (i % self.total_centers) == self.center:
                    client_dataset.extend(gd)


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

        print(set([f[-1] for f in client_dataset]))

        client_df = pd.DataFrame.from_records(client_dataset,
                        columns =['filename', 'image_id', self.label_type])

        return client_df


    def __len__(self):
        return len(self.df2)

    def __getitem__(self, idx):
        image_name = self.df2["filename"].iloc[idx]
        with Image.open(os.path.join(self.images_root, image_name )) as pilimg:
            image  = self.transforms(pilimg)
        target = self.df2[self.label_type].iloc[idx]

        return image, target

    def _remap_values(self, value, max_value=100): #100 for Google FedVision dataset
        return int(value // (max_value/self.total_centers))



###=================== Getter Functions ========================================


def getCifarCLSLoaders(cfg, center_index = None, override_csv = None):
    """ center_index: None/all --> pooled
    """
    if center_index == None: center_index = "all"

    data_path     = cfg.data_root_path
    dataset_type  = cfg.dataset_type
    label_type    = cfg.label_type
    info_log_path = cfg.gLogPath +'/misc_data.txt'
    img_size_in   = cfg.image_size
    batch_size    = cfg.batch_size
    workers       = cfg.workers
    alpha         = cfg.dirichlet_alpha
    iid_ness      = cfg.iid_ness
    total_centers = cfg.data_centers_count
    num_classes   = cfg.clsfy_layers[-1] #100
    override_csv = cfg.override_csv if cfg.override_csv else None


    traindataset = Cifar_JFedDataset( data_path= data_path, csv_name=override_csv,
                    dataset_type=dataset_type,
                    center= center_index, split_type = "cls_train",
                    dirichlet_alpha = alpha, iid_ness=iid_ness,
                    total_centers=total_centers,
                    label_type=label_type,
                    transforms=CifarClassifyAuguments(method="train",
                                                      image_size=img_size_in))

    validdataset = Cifar_JFedDataset( data_path= data_path, csv_name=override_csv,
                    dataset_type=dataset_type,
                    center= center_index, split_type = "cls_valid",
                    dirichlet_alpha = alpha,  iid_ness=iid_ness,
                    total_centers=total_centers,
                    label_type=label_type,
                    transforms=CifarClassifyAuguments(method="infer",
                                                      image_size=img_size_in))

    # class_weights = get_class_weights(traindataset.targets, nclasses=num_classes)

    trainloader  = torch.utils.data.DataLoader( traindataset, shuffle=True,
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=True, persistent_workers=False)

    validloader  = torch.utils.data.DataLoader( validdataset, shuffle=False,
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=True)

    lutl.LOG2DICTXT({"DC":("CIFAR", center_index), "Train-":len(traindataset),
                    "TargetClasses": str(set(traindataset.targets)),
                    "Transform": str(traindataset.transforms.get_composition()),
                    ## "class-weights":str(class_weights)
                     }, info_log_path)

    lutl.LOG2DICTXT({"DC":("CIFAR", center_index), "Valid-":len(validdataset),
                    "TargetClasses": str(set(validdataset.targets)),
                    "Transform": str(validdataset.transforms.get_composition()),
                    }, info_log_path)

    if override_csv:
        lutl.LOG2TXT(f"OverRide CSV-set: {override_csv} !^!^!^!", info_log_path)

    return trainloader, validloader



def getCifarTESTLoader(cfg, center_index = None):
    """ center_index: None/all --> pooled
    """
    if center_index == None: center_index = "all"

    data_path     = cfg.data_root_path
    dataset_type  = cfg.dataset_type
    label_type    = cfg.label_type
    info_log_path = cfg.gLogPath +'/misc_data.txt'
    img_size_in   = cfg.image_size
    batch_size    = cfg.batch_size
    workers       = cfg.workers

    dataset = Cifar_JFedDataset( data_path= data_path,
                    dataset_type=dataset_type,
                    center= center_index, split_type = "test",
                    label_type=label_type,
                    transforms=CifarClassifyAuguments(method="infer",
                                                      image_size=img_size_in))

    testloader  = torch.utils.data.DataLoader(dataset, shuffle=False,
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=True)

    lutl.LOG2DICTXT({"DC":("CIFAR", center_index), "TEST-":len(dataset),
                    "TargetClasses": str(set(dataset.targets)),
                    "Transform": str(dataset.transforms.get_composition()),
                     }, info_log_path)

    return testloader



def getCifarSSLLoader(cfg, center_index = None, ssl_transforms=None):
    """ center_index: None/all --> pooled
    """
    if center_index == None: center_index = "all"

    data_path     = cfg.data_root_path
    dataset_type  = cfg.dataset_type
    label_type    = cfg.label_type
    info_log_path = cfg.gLogPath +'/misc_data.txt'
    batch_size    = cfg.batch_size
    img_size_in   = cfg.image_size
    workers       = cfg.workers
    alpha         = cfg.dirichlet_alpha
    iid_ness      = cfg.iid_ness
    total_centers = cfg.data_centers_count

    dataset = Cifar_JFedDataset( data_path= data_path,
                    dataset_type=dataset_type,
                    center= center_index, split_type = "ssl_train",
                    dirichlet_alpha = alpha,  iid_ness=iid_ness,
                    total_centers=total_centers,
                    label_type=label_type,
                    transforms=ssl_transforms)

    loader  = torch.utils.data.DataLoader(dataset, shuffle=True,
                        batch_size=batch_size, num_workers=workers,
                        drop_last=True, ## Important
                        pin_memory=True)

    lutl.LOG2DICTXT({"DC":("CIFAR", center_index), "SSL-":len(dataset),
                     "Transform": str(dataset.transforms.get_composition()),
                     }, info_log_path)

    return loader



## ============================================================================

if __name__ == "__main__":

    dataset = Cifar_JFedDataset( data_path="/home/joseph.benjamin/WERK/fed-cvpr/data/cifar100-jfed",
                                  dataset_type="cifar100",
                                  center=5, split_type="cls_train",
                                  transforms=None)
    for i in range(10):
        a, b = dataset.__getitem__(i)
        print(b)