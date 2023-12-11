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
from sketching.race_sketch import RaceSketchVec


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

    elif cfg.dataset ==  "MNIST":
        from datacode.humble_jfed_data import MNISTkind_JFedDatset
        from datacode.augmentations import HumbleTransforms

        traindataset = MNISTkind_JFedDatset(data_path  = cfg.data_path,
                                    dataset_type  ='MNIST',
                                    split_type    = split_type,
                                    center        = center_index,
                                    total_centers = cfg.data_centers_count,
                                    iid_ness      = cfg.iid_ness,
                                    semi_client_per_class = 2,
                                    transform=HumbleTransforms
                                    )

    else:
        raise ValueError(f"Unsupported data type specfied {cfg.dataset}")

    trainloader  = torch.utils.data.DataLoader( traindataset,
                        shuffle=False,
                        batch_size=1,
                        num_workers=2,
                        pin_memory=True)

    return trainloader

### ============================================================================

def pretrained_weight_loader(model, weight_path):
    pth_wgt_state = torch.load(weight_path)

    new_state_dict = {}
    for key, value in pth_wgt_state.items():
        new_key = key.replace("backbone.", "")
        new_state_dict[new_key] = value

    ret_msg = model.load_state_dict(new_state_dict, strict=False)
    print(ret_msg)
    return model


class ModelFeatureMean():
    def __init__(self, weight_path=None ,device="cuda") -> None:
        self.model = torchvision.models.resnet18(weights="DEFAULT")
        self.model = self.model.to(device)
        self.model.fc = torch.nn.Identity()

        if weight_path: self.model = pretrained_weight_loader(self.model, weight_path)

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


class ModelFeatureCovariance():
    def __init__(self, weight_path=None ,device="cuda") -> None:
        self.model = torchvision.models.resnet18(weights="DEFAULT")
        self.model = self.model.to(device)
        self.model.fc = torch.nn.Identity()

        if weight_path: self.model = pretrained_weight_loader(self.model, weight_path)

        self.model.eval()
        self.counter = 0
        self.featp_holder = []

    def accum_one_sample(self,x):
        featp = self.model.forward(x)
        self.featp_holder.append(featp.flatten())
        self.counter+=1

    def get_full_aggregate(self):

        num_features = 512
        featp_NxD = torch.stack(self.featp_holder)
        print(featp_NxD.shape)

        covariance = torch.cov(featp_NxD.T) #torch.cov rows are the variables and columns are the observations
        print(covariance.shape)

        res = covariance
        return res.detach().cpu().numpy()



class Image_Histogram():
    def __init__(self, img_size,
                 channels = 3,
                 device="cuda") -> None:
        self.channels = channels
        self.R_hist = torch.zeros(256, dtype=torch.float32).to(device)
        self.G_hist = torch.zeros(256, dtype=torch.float32).to(device)
        self.B_hist = torch.zeros(256, dtype=torch.float32).to(device)
        self.counter = 0
        assert (channels == 3), "channels other than 3 logic Not handled"

    def accum_one_sample(self,x):
        R_hist = torch.histc(x[:,0,:,:].flatten(), bins=256, min=-1, max=1)
        G_hist = torch.histc(x[:,1,:,:].flatten(), bins=256, min=-1, max=1)
        B_hist = torch.histc(x[:,2,:,:].flatten(), bins=256, min=-1, max=1)
        self.R_hist+= R_hist
        self.G_hist+= G_hist
        self.B_hist+= B_hist
        self.counter +=1

    def get_full_aggregate(self):
        res = torch.stack([self.R_hist,self.G_hist, self.B_hist])
        return res.detach().cpu().numpy()



##------------------------------------------------------------------------------
class Image_CountSketching():
    def __init__(self, img_size,
                 channels = 3,
                 device="cuda") -> None:

        cols = 2 ** ( int.bit_length(img_size**2 - 1) -2)
        self.csobj = CountSketchVec(d=img_size*img_size*channels, c=cols, r=16, device=device)
        self.counter = 0

    def accum_one_sample(self,x):
        x = x.flatten()
        self.csobj.accumulateVec(x)
        self.counter+=1

    def get_full_aggregate(self):
        res = self.csobj.table
        return res.detach().cpu().numpy()


class Feature_CountSketching():

    def __init__(self, weight_path=None, device="cuda") -> None:

        self.model = torchvision.models.resnet18(weights="DEFAULT")
        self.model = self.model.to(device)
        self.model.fc = torch.nn.Identity()

        if weight_path: self.model = pretrained_weight_loader(self.model, weight_path)

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



### ============================================================================


def getSketcher(cfg):
    img_size = cfg.image_size
    channels = 3 if not cfg.channels else cfg.channels
    sketcher_dict = {
        "feature-only": ModelFeatureMean(weight_path=cfg.weight_path,
                                        device="cuda"),
        "feature-Covar": ModelFeatureCovariance(weight_path=cfg.weight_path,
                                device="cuda"),
        "feature-CountSketch": Feature_CountSketching(weight_path=cfg.weight_path,
                                                      device="cuda"),

        "image-Histogram" : Image_Histogram(img_size,  channels=channels,device="cuda"),
        "image-CountSketch": Image_CountSketching(img_size, channels=channels,device="cuda"),

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
        laoder_list = []
        trainloader = getAdataloader(cfg, center_index, split_type="cls_train")
        laoder_list.append(trainloader)

        if not cfg.disable_validation:
            validloader = getAdataloader(cfg, center_index, split_type="cls_valid")
            laoder_list.append(validloader)

        ## Load Weights
        if cfg.weight_root_path:
            cfg.weight_path = f"{cfg.weight_root_path}/center_{center_index}/weights/bestmodel.pth"

        sketcher = getSketcher(cfg)

        with torch.no_grad():
            for laoder in laoder_list:
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


def run_for_mnist():
    CFG.dataset   = "MNIST"
    CFG.data_path = "/home/joseph.benjamin/WERK/fed-cvpr/data/torch-data/"
    CFG.data_centers_count = 5
    CFG.image_size = 28
    CFG.channels   = 1
    CFG.disable_validation = True

    for alp in ["non", "full", "semi"]:
            CFG.iid_ness = alp
            data_sketch_main(CFG, file_suffix=f"{alp}-")



if __name__ == '__main__':
    CFG.weight_root_path = None# "hypotheses/Cls1-isic/E00-Cls-Baseline-01/"

    # CFG.sketch_method = "feature-only"
    CFG.sketch_method = "feature-Covar"

    # CFG.sketch_method = "feature-CountSketch"
    # CFG.sketch_method = "image-CountSketch"
    # CFG.sketch_method = "image-Histogram"
    # CFG.sketch_method = "imagepix-RaceSketch"

    # run_for_mnist()
    run_for_isicflamby()



