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
from datacode.augmentations import IsicClassifyAuguments
from utilities.metricUtils import get_class_weights


## ======================== Dataset Class ======================================

class Isic2019_JFedDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_path: str = None, #path to dataset images
        csv_name = None, # for some special data partitioning
        center = "all",  # all--> Pooled
        split_type: str = "ssl_train", # cls_train / cls_valid / test
        transforms=None,
    ):

        # cls_csv = csv_name if csv_name else "isic_clsfy_data.csv"
        cls_csv = csv_name if csv_name else "isic_Full_Train_data.csv"
        ssl_csv = csv_name if csv_name else  "isic_ssl_data.csv"
        test_csv = csv_name if csv_name else "isic_test_data.csv"

        if data_path:
            if not (os.path.exists(data_path)):
                raise ValueError(f"The string {data_path} is not a valid path.")
            self.images_root = os.path.join(data_path, "ISIC_2019_Training_Input_preprocessed")
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
            df2 = pd.read_csv(os.path.join(data_path, test_csv))
        else: raise("Unknown Split type specified ....", split_type)



        self.center = center
        self.pooled = True if center == "all" else False

        if not self.pooled:
            df2 = df2[df2["center"] == center]

        self.df2     = df2.reset_index()
        self.targets = list(self.df2["target"])

        self.transforms = transforms if transforms else torch_transforms.ToTensor()


    def __len__(self):
        return len(self.df2)

    def __getitem__(self, idx):
        image_name = self.df2["image"].iloc[idx]
        with Image.open(os.path.join(self.images_root, image_name + ".jpg")) as pilimg:
            image = self.transforms(pilimg)
        target = self.df2["target"].iloc[idx]

        return image, target



###=================== Getter Functions ========================================


def default_Flamby_Isic_dataset():
    """ default flamby datset class; check `data-procs` folder for more
        details on processing
    """
    pass
    # from flamby.datasets.fed_isic2019 import FedIsic2019

    # ## Augumentations default in ISIC-FLamby
    # traindataset = FedIsic2019(pooled=True, train=True)
    # validdataset = FedIsic2019(pooled=True, train=False)


def getIsicCLSLoaders(cfg, center_index = None):
    """ center_index: None/all --> pooled
    """
    if center_index == None: center_index = "all"

    data_path = cfg.data_root_path
    info_log_path = cfg.gLogPath +'/misc_data.txt'
    batch_size = cfg.batch_size
    workers   = cfg.workers
    num_classes = cfg.clsfy_layers[-1]
    override_csv = cfg.override_csv if cfg.override_csv else None


    traindataset = Isic2019_JFedDataset( data_path= data_path, csv_name=override_csv,
                    center= center_index, split_type = "cls_train",
                    transforms=IsicClassifyAuguments(method="kaggle"))

    validdataset = Isic2019_JFedDataset( data_path= data_path, csv_name=override_csv,
                    center= center_index, split_type = "cls_valid",
                    transforms=IsicClassifyAuguments(method="infer"))

    class_weights = get_class_weights(traindataset.targets, nclasses=num_classes)

    trainloader  = torch.utils.data.DataLoader( traindataset, shuffle=True,
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=True)

    validloader  = torch.utils.data.DataLoader( validdataset, shuffle=False,
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=True)

    lutl.LOG2DICTXT({"DC":("ISIC", center_index), "Train-":len(traindataset),
                     "class-weights":str(class_weights)},
                     info_log_path)
    lutl.LOG2DICTXT({"DC":("ISIC", center_index), "Valid-":len(validdataset)}, info_log_path)

    if override_csv:
        lutl.LOG2TXT(f"OverRide CSV-set: {override_csv} !^!^!^!", info_log_path)

    return trainloader, validloader



def getIsicTESTLoader(cfg, center_index = None):
    """ center_index: None/all --> pooled
    """
    if center_index == None: center_index = "all"

    data_path = cfg.data_root_path
    info_log_path = cfg.gLogPath +'/misc_data.txt'
    batch_size = cfg.batch_size
    workers    = cfg.workers
    image_size = cfg.image_size

    dataset = Isic2019_JFedDataset( data_path= data_path,
                    center= center_index, split_type = "test",
                    transforms=IsicClassifyAuguments(method="infer"))

    testloader  = torch.utils.data.DataLoader(dataset, shuffle=False,
                        batch_size=batch_size, num_workers=workers,
                        pin_memory=True)

    lutl.LOG2DICTXT({"DC":("ISIC", center_index),
                     "TEST-":len(dataset)}, info_log_path)

    return testloader



def getIsicSSLLoader(cfg, center_index = None, ssl_transforms= None):
    """ center_index: None/all --> pooled
    """
    if center_index == None: center_index = "all"

    data_path = cfg.data_root_path
    info_log_path = cfg.gLogPath +'/misc_data.txt'
    batch_size = cfg.batch_size
    workers   = cfg.workers

    dataset = Isic2019_JFedDataset( data_path= data_path,
                    center= center_index, split_type = "ssl_train",
                    transforms=ssl_transforms)

    loader  = torch.utils.data.DataLoader(dataset, shuffle=True,
                        batch_size=batch_size, num_workers=workers,
                        drop_last=True, ## Important
                        # prefetch_factor=1,
                        pin_memory=True)

    lutl.LOG2DICTXT({"DC":("ISIC", center_index),
                     "SSL-":len(dataset)}, info_log_path)

    return loader



## ============================================================================

if __name__ == "__main__":

    dataset = Isic2019_JFedDataset( data_path="/home/joseph.benjamin/WERK/fed-cvpr/data/isic2019-jfed",
                                  center=5, split_type="cls_train",
                                  transforms=None)
    for i in range(10):
        a, b = dataset.__getitem__(i)
        print(b)