import os, sys
import time, datetime
import argparse, json
import h5py

import numpy as np
import torch
import torchvision

from tqdm.autonotebook import tqdm

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl

from sketching.count_sketch import CountSketchVec


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")

##============================= Configure and Setup ============================

CFG = rutl.ObjDict(

checkpoint_dir= "hypotheses/DataSketchColl/trail-001/",
)

### ----------------------------------------------------------------------------

def getAdataloader(cfg, center_index, split_type="cls_train"):

    if cfg.dataset == "ISIC":
        from datacode.augmentations import IsicClassifyAuguments
        from datacode.isic_jfed_data import Isic2019_JFedDataset
        img_size_in = cfg.image_size
        traindataset = Isic2019_JFedDataset( data_path= cfg.data_path,
                        csv_name='isic_Full_Train_data.csv',
                        center= center_index,
                        split_type = split_type,
                        transforms=IsicClassifyAuguments(method="infer",
                                                         image_size=img_size_in))

    elif cfg.dataset == "CIFAR100":
        from datacode.augmentations import CifarClassifyAuguments
        from datacode.cifar100_jfed_data import Cifar100_JFedDataset
        img_size_in   = cfg.image_size

        traindataset = Cifar100_JFedDataset( data_path= cfg.data_path,
                        csv_name="cifar100_Full_Train_data.csv",
                        center= center_index,
                        split_type = split_type,
                        dirichlet_alpha = cfg.dirichlet_alpha,
                        total_centers=cfg.data_centers_count,
                        transforms=CifarClassifyAuguments(method="infer",
                                                        image_size=img_size_in))
    elif cfg.dataset == "ORGAN-MNIST":
        from datacode.augmentations import OrganMnistClassifyAuguments
        from datacode.orgmnist_jfed_data import OrganMnist_JFedDataset
        img_size_in   = cfg.image_size
        traindataset = OrganMnist_JFedDataset( data_path= cfg.data_path,
                        csv_name="organmnist_Full_Train_data.csv",
                        center= center_index,
                        split_type = split_type,
                        transforms=OrganMnistClassifyAuguments(method="infer",
                                                        image_size=img_size_in))
    else:
        raise ValueError(f"Unsupported data type specfied {cfg.dataset}")

    trainloader  = torch.utils.data.DataLoader( traindataset,
                        shuffle=False,
                        batch_size=1,
                        num_workers=2,
                        pin_memory=True)

    return trainloader

### ============================================================================

class ModelFeatureAgg():
    def __init__(self, device="cuda") -> None:
        self.model = torchvision.models.resnet18(weights="DEFAULT")
        self.model = self.model.to(device)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.counter = 0
        self.featp_holder = torch.zeros(512).to(device)

    def accum_one_sample(self,x):
        featp = self.model.forward(x)
        self.featp_holder += featp.flatten()
        self.counter+=1

    def get_full_aggregate(self):
        res = self.featp_holder / self.counter
        return res.detach().cpu().numpy()

class ImageSketching():
    def __init__(self, img_size, device="cuda") -> None:

        self.csobj = CountSketchVec(d=img_size*img_size *3, c=2**13, r=16, device=device)
        self.counter = 0

    def accum_one_sample(self,x):
        self.csobj.accumulateVec(x.flatten())
        self.counter+=1

    def get_full_aggregate(self):
        res = self.csobj.table
        return res.detach().cpu().numpy()


class FeatureSketching():

    def __init__(self, img_size, device="cuda") -> None:

        self.model = torchvision.models.resnet18(weights="DEFAULT")
        self.model = self.model.to(device)
        self.model.fc = torch.nn.Identity()
        self.model.eval()

        self.csobj = CountSketchVec(d=512, c=32, r=16, device=device)
        self.counter = 0

    def accum_one_sample(self,x):
        featp = self.model.forward(x)
        self.csobj.accumulateVec(featp.flatten())
        self.counter+=1

    def get_full_aggregate(self):
        res = self.csobj.table
        return res.detach().cpu().numpy()


### ----------------------------------------------------------------------------

def getSketcher(cfg):
    img_size = cfg.image_size
    sketcher_dict = {
        "feature-only": ModelFeatureAgg(),
        "image-sketch": ImageSketching(img_size, device="cuda"),
        "feature-sketch": FeatureSketching(img_size, device="cuda"),
    }
    return sketcher_dict[cfg.sketch_method]





def data_sketch_main(cfg, file_suffix=""):

    rutl.START_SEED()
    gpu_device = torch.device("cuda")
    torch.cuda.device(gpu_device)
    print("GPU device", gpu_device)

    sfolderpath = cfg.checkpoint_dir+"/"+ cfg.sketch_method +"/"
    os.makedirs(sfolderpath, exist_ok=True)

    h5path = f"{sfolderpath}/{cfg.dataset}-C{cfg.data_centers_count}-{file_suffix}.h5"
    h5file = h5py.File(h5path,"w")

    center_list = list(range(cfg.data_centers_count))+["all"]

    for center_index in center_list:
        trainloader = getAdataloader(cfg, center_index, split_type="cls_train")
        validloader = getAdataloader(cfg, center_index, split_type="cls_valid")

        sketcher = getSketcher(cfg)

        with torch.no_grad():
            for laoder in [trainloader, validloader]:
                for img, _ in tqdm(laoder):
                    img = img.to(gpu_device, non_blocking=True)
                    sketcher.accum_one_sample(img)

        h5file.create_dataset(str(center_index), data=sketcher.get_full_aggregate(),
                              dtype=float)
        del sketcher

    h5file.close()
    return h5path


### ============================================================================



def run_for_cifar100():
    CFG.dataset   = "CIFAR100"
    CFG.data_path = "/home/joseph.benjamin/WERK/fed-cvpr/data/cifar100-jfed/"
    CFG.data_centers_count = 10
    CFG.image_size = 224

    for alp in [1000, 100, 10, 1, 0, 0.5]:
            CFG.dirichlet_alpha = alp
            data_sketch_main(CFG, file_suffix=f"{alp}-")



def run_for_isicflamby():
    CFG.dataset   = "ISIC"
    CFG.data_path = "/home/joseph.benjamin/WERK/fed-cvpr/data/isic2019-jfed/"
    CFG.data_centers_count = 6
    CFG.image_size = 200

    data_sketch_main(CFG)


def run_for_organmnist():
    CFG.dataset   = "ORGAN-MNIST"
    CFG.data_path = "/home/joseph.benjamin/WERK/fed-cvpr/data/organmnist-jfed"
    CFG.data_centers_count = 3
    CFG.image_size = 224

    data_sketch_main(CFG)



if __name__ == '__main__':
    CFG.sketch_method = "feature-only"
    # CFG.sketch_method = "feature-sketch"
    # CFG.sketch_method = "image-sketch"

    run_for_isicflamby()



