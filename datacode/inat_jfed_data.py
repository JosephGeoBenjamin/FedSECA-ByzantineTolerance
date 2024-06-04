import time
import os, sys, pathlib
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter

import torch
import torchvision.transforms as torch_transforms
from torchvision.datasets.folder import default_loader as tv_image_loader
from torch.utils.data import ConcatDataset

sys.path.append(os.getcwd())
import utilities.logUtils as lutl
from datacode.augmentations import DeiTAuguments
from utilities.metricUtils import get_class_weights


## ======================== Dataset Class ======================================

## dataset_type : No. Clients
# geo_30k : 11
# geo_10k : 38
# geo_3k  : 135
# geo_1k  : 368
# geo_300 : 1208
# geo_100 : 3606


class INaturalist2017_JFedDataset(torch.utils.data.Dataset):
    "Based on iNat2017 for Google Vision Federated"
    def __init__(
        self, data_path: str = None, #path to dataset images
        csv_name = None, # for some special data partitioning
        dataset_type = "geo_30k",
        center = "all",  # all--> Pooled
        split_type: str = "cls_train", # cls_train / cls_valid / test / ssl_train
        label_type = "class", # target -> 11 class; target_grouped -> 7 classes
        transforms=None,
    ):

        cls_csv = csv_name if csv_name else  f"inat_{dataset_type}_train_data_noval.csv"
        ssl_csv = csv_name if csv_name else  "NoFileExists_thisisplaceholder"
        test_csv = csv_name if csv_name else "inat_geo_test_data.csv"

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
        else: raise("Unknown Split type specified ....", split_type)

        if len(self.images_root) ==0: raise ("No images path read, plase check path and folder name `train_images/test_images`")

        self.total_centers = df2["center"].nunique() if "center" in df2 else 1
        self.center = center
        self.label_type = label_type
        self.pooled = True if center == "all" else False

        if not self.pooled:
            df2 = df2[df2["center"] == center]

        self.df2     = df2.reset_index()
        self.targets = list(self.df2[label_type])
        self.transforms = transforms if transforms else torch_transforms.ToTensor()


    def __len__(self):
        return len(self.df2)

    def __getitem__(self, idx):
        image_name = self.df2["filename"].iloc[idx]
        pilimg = Image.open(os.path.join(self.images_root, image_name )).convert("RGB")
        image  = self.transforms(pilimg)
        target = self.df2[self.label_type].iloc[idx]

        return image, target



###=================== Getter Functions ========================================


def getINaturalistCLSLoaders(cfg, center_index = None, override_csv = None):
    """ center_index: None/all --> pooled
    """
    if center_index == None: center_index = "all"

    data_path     = cfg.data_root_path
    info_log_path = cfg.gLogPath +'/misc_data.txt'
    img_size_in   = cfg.image_size
    batch_size    = cfg.batch_size
    workers       = cfg.workers
    num_classes   = cfg.clsfy_layers[-1] #100
    override_csv = cfg.override_csv if cfg.override_csv else None


    traindataset = INaturalist2017_JFedDataset( data_path= data_path, csv_name=override_csv,
                    center= center_index, split_type = "cls_train",
                    transforms=DeiTAuguments(method="train",
                                            image_size=img_size_in))

    validdataset = INaturalist2017_JFedDataset( data_path= data_path, csv_name=override_csv,
                    center= center_index, split_type = "cls_valid",
                    transforms=DeiTAuguments(method="infer",
                                            image_size=img_size_in))

    # class_weights = get_class_weights(traindataset.targets, nclasses=num_classes)

    trainloader  = torch.utils.data.DataLoader( traindataset, shuffle=True,
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=True, persistent_workers=False)

    validloader  = torch.utils.data.DataLoader( validdataset, shuffle=False,
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=True)

    lutl.LOG2DICTXT({"DC":("iNaturalist", center_index), "Train-":len(traindataset),
                     "Transform": str(traindataset.transforms.get_composition()),
                    #  "class-weights":str(class_weights)
                     }, info_log_path)
    lutl.LOG2DICTXT({"DC":("iNaturalist", center_index), "Valid-":len(validdataset),
                     "Transform": str(validdataset.transforms.get_composition()),
                     }, info_log_path)

    if override_csv:
        lutl.LOG2TXT(f"OverRide CSV-set: {override_csv} !^!^!^!", info_log_path)

    return trainloader, validloader



def getINaturalistTESTLoader(cfg, center_index = None):
    """ center_index: None/all --> pooled
    """
    if center_index == None: center_index = "all"

    data_path = cfg.data_root_path
    info_log_path = cfg.gLogPath +'/misc_data.txt'
    img_size_in   = cfg.image_size
    batch_size = cfg.batch_size
    workers    = cfg.workers

    dataset = INaturalist2017_JFedDataset( data_path= data_path,
                    center= center_index, split_type = "test",
                    transforms=DeiTAuguments(method="infer",
                                            image_size=img_size_in))

    testloader  = torch.utils.data.DataLoader(dataset, shuffle=False,
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=True)

    lutl.LOG2DICTXT({"DC":("iNaturalist", center_index), "TEST-":len(dataset),
                    "Transform": str(dataset.transforms.get_composition()),
                     }, info_log_path)

    return testloader



def getINaturalistSSLLoader(cfg, center_index = None, ssl_transforms=None):
    """ center_index: None/all --> pooled
    """
    if center_index == None: center_index = "all"

    data_path     = cfg.data_root_path
    info_log_path = cfg.gLogPath +'/misc_data.txt'
    batch_size    = cfg.batch_size
    img_size_in   = cfg.image_size
    workers       = cfg.workers
    total_centers = cfg.data_centers_count

    dataset = INaturalist2017_JFedDataset( data_path= data_path,
                    center= center_index, split_type = "ssl_train",
                    transforms=ssl_transforms)

    loader  = torch.utils.data.DataLoader(dataset, shuffle=True,
                        batch_size=batch_size, num_workers=workers,
                        drop_last=True, ## Important
                        pin_memory=True)

    lutl.LOG2DICTXT({"DC":("iNaturalist", center_index), "SSL-":len(dataset),
                     "Transform": str(dataset.transforms.get_composition()),
                     }, info_log_path)

    return loader


## ============================================================================

if __name__ == "__main__":

    dataset = INaturalist2017_JFedDataset( data_path="/home/joseph.benjamin/WERK/fed-cvpr/data/inat2017-jfed",
                                  center=5, split_type="cls_train",
                                  transforms=None)
    for i in range(10):
        a, b = dataset.__getitem__(i)
        print(b)