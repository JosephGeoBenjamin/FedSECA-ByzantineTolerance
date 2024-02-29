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

from sklearn.metrics.pairwise import cosine_similarity

print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")

##============================= Configure and Setup ============================

os.environ['CUDA_PATH'] = '/home/joseph.benjamin/.conda/envs/sfed/' #for sinkhorn
from sketching.sinkhorn_metric import sinkhorn_pointcloud_pytorch


CFG = rutl.ObjDict(
    cloud_point_count = 1000,
    checkpoint_dir= "hypotheses/DataSketchColl/Cifar10-main-002/1K-pts/",
)

### ----------------------------------------------------------------------------

def getAdataloader(cfg, center_index, split_type="cls_train"):

    if cfg.dataset == "ISIC":
        from datacode.augmentations import IsicClassifyAuguments
        from datacode.isic_jfed_data import Isic2019_JFedDataset
        img_size_in = cfg.image_size
        traindataset = Isic2019_JFedDataset( data_path= cfg.data_path,
                        csv_name='isic_train_data.csv',
                        center= center_index,
                        split_type = split_type,
                        transforms=IsicClassifyAuguments(method="infer",
                                                         image_size=img_size_in))

    elif cfg.dataset == "CIFAR":
        from datacode.augmentations import CifarClassifyAuguments
        from datacode.cifar_jfed_data import Cifar_JFedDataset
        img_size_in   = cfg.image_size
        traindataset = Cifar_JFedDataset( data_path= cfg.data_path,
                        csv_name = cfg.csv_name,
                        dataset_type = cfg.dataset_type,
                        center = center_index,
                        split_type = split_type,
                        dirichlet_alpha = cfg.dirichlet_alpha,
                        iid_ness = cfg.iid_ness,
                        label_type = cfg.label_type,
                        total_centers=cfg.data_centers_count,
                        transforms=CifarClassifyAuguments(method="infer",
                                                        image_size=img_size_in))
    elif cfg.dataset == "ORGANMNIST":
        from datacode.augmentations import OrganMnistClassifyAuguments
        from datacode.orgmnist_jfed_data import OrganMnist_JFedDataset
        img_size_in   = cfg.image_size
        traindataset = OrganMnist_JFedDataset( data_path= cfg.data_path,
                        csv_name="organmnist_trainV2_data.csv",
                        center= center_index,
                        split_type = split_type,
                        transforms=OrganMnistClassifyAuguments(method="infer",
                                                        image_size=img_size_in))

    elif cfg.dataset ==  "HUMBLE":
        from datacode.humble_jfed_data import MNISTkind_JFedDatset
        from datacode.augmentations import HumbleAuguments

        traindataset = MNISTkind_JFedDatset(data_path  = cfg.data_path,
                                    dataset_type  = cfg.dataset_type,
                                    split_type    = split_type,
                                    center        = center_index,
                                    total_centers = cfg.data_centers_count,
                                    iid_ness      = cfg.iid_ness,
                                    semi_client_per_class = 2,
                                    transforms=HumbleAuguments()
                                    )

    else:
        raise ValueError(f"Unsupported data type specfied {cfg.dataset}")

    trainloader  = torch.utils.data.DataLoader( traindataset,
                        shuffle=False,
                        batch_size=1,
                        num_workers=2,
                        pin_memory=True)

    return trainloader

def getModel(model_name):
    model_name = "resnet18" if not model_name else model_name

    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights="DEFAULT")
        model.fc = torch.nn.Identity()
    elif model_name == "resnet9":
        from algorithms.feature_extractor import resnet9
        model = resnet9()
        model.classifier = torch.nn.Identity()
    if model_name == 'efficientnet_b0':
        model = torchvision.models.efficientnet_b0(zero_init_residual=True,
                                weights="DEFAULT")
        model.classifier = torch.nn.Identity()  #remove fc of default arch

    return model



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


def tensor_cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)
    https://github.com/pytorch/pytorch/issues/19037#issuecomment-814496788
    """
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    bias_corrector = int(not bool(bias) and bool(tensor.shape[-1]-1) ) #hack to fix for single sample case
    factor = 1 / (tensor.shape[-1] - bias_corrector)
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def min_max_scale(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data


def find_orthogonal_vector_numpy(v_matrix, dim=1):
    """ takes NxD matrix
    """
    # Create a random matrix of the same shape as v_matrix
    random_vecs = np.random.rand(*v_matrix.shape)

    # Project the random matrix onto v_matrix along the specified dimension
    # equivalent => projection = np.dot(random_vector, v) / np.dot(v, v) * v
    projection = np.sum(random_vecs * v_matrix, axis=dim, keepdims=True) \
        / np.sum(v_matrix**2, axis=dim, keepdims=True) * v_matrix

    # Subtract the projection from the random matrix to get an orthogonal matrix
    orthogonal_vecs = random_vecs - projection

    return orthogonal_vecs


class ModelFeatureMean():
    def __init__(self, model_name ,weight_path=None ,device="cuda") -> None:
        self.model = getModel(model_name)
        self.model = self.model.to(device)

        if weight_path: self.model = pretrained_weight_loader(self.model, weight_path)

        self.model.eval()
        self.counter = 0
        self.featp_holder = torch.zeros(512).to(device)

    def accum_one_sample(self,x, _):
        featp = self.model.forward(x)
        self.featp_holder += featp.flatten()
        self.counter+=1

    def get_full_aggregate(self):
        res = self.featp_holder / self.counter
        return res.detach().cpu().numpy()


class ModelFeatureCovariance():
    def __init__(self, model_name, weight_path=None ,device="cuda") -> None:
        self.model = getModel(model_name)
        self.model = self.model.to(device)
        self.model.fc = torch.nn.Identity()

        if weight_path: self.model = pretrained_weight_loader(self.model, weight_path)

        self.model.eval()
        self.counter = 0
        self.featp_holder = []

    def accum_one_sample(self,x, _):
        featp = self.model.forward(x)
        self.featp_holder.append(featp.flatten())
        self.counter+=1

    def get_full_aggregate(self):

        featp_NxD = torch.stack(self.featp_holder)
        print(featp_NxD.shape)

        covariance = torch.cov(featp_NxD.T) #torch.cov rows are the variables and columns are the observations
        print(covariance.shape)

        res = covariance
        return res.detach().cpu().numpy()




class ModelFeatureGaussDistribution():
    def __init__(self, model_name, total_points=500, weight_path=None, classwise=False ,device="cuda") -> None:
        self.model = getModel(model_name)
        self.model = self.model.to(device)
        self.model.fc = torch.nn.Identity()

        self.total_points = total_points

        if weight_path: self.model = pretrained_weight_loader(self.model, weight_path)

        self.model.eval()
        self.classwise = classwise #This will make it Class-based GMMs
        self.counter = 0
        self.full_featp_holder = []
        self.class_featp_holder = {}


    def _add_feature_to_dict(self, z, c):
        c = c.item()
        if c in self.class_featp_holder.keys():
            self.class_featp_holder[c].append(z)
        else:
            self.class_featp_holder[c] = [z]


    def accum_one_sample(self, x, c):
        featp = self.model.forward(x)
        if self.classwise: self._add_feature_to_dict(featp.flatten(), c)
        else: self.full_featp_holder.append(featp.flatten())

        self.counter+=1


    def _find_mean_cov(self,featp_list):

        featp_NxD = torch.stack(featp_list)
        print(featp_NxD.shape)

        mean = torch.mean(featp_NxD, dim=0)
        cov = tensor_cov(featp_NxD.T)  #torch.cov rows are the variables and columns are the observations
        print(mean.shape, cov.shape)
        return mean.detach().cpu().numpy(), cov.detach().cpu().numpy()


    def get_full_aggregate(self):

        if self.classwise:
            mean_list = []; cov_list = []; walpha_list = []
            for c in self.class_featp_holder:
                c_mean, c_cov = self._find_mean_cov(self.class_featp_holder[c])
                c_walpha = len(self.class_featp_holder[c]) / self.counter
                cov_list.append(c_cov)
                mean_list.append(c_mean)
                walpha_list.append(c_walpha)
            data_points = self._generate_mixture_of_gaussians(mean_list, cov_list, walpha_list, self.total_points)

        else:
            f_mean, f_cov = self._find_mean_cov(self.full_featp_holder)
            data_points = np.random.multivariate_normal(f_mean, f_cov, self.total_points)

        return data_points

    def _generate_mixture_of_gaussians(self,  means_list, covs_list, walphas_list, num_points):

        num_components = len(means_list)
        component_indices = np.random.choice(num_components, size=num_points, p=walphas_list)

        data_points = np.zeros((num_points, len(means_list[0])))

        for i in range(num_components):
            component_mask = (component_indices == i)
            num_samples = np.sum(component_mask)
            data_points[component_mask] = np.random.multivariate_normal(
                means_list[i], covs_list[i], num_samples)

        return data_points


    @classmethod
    def compute_distance_matrix(cls, hdf5_file_path):

        print("Computing Distance Matrix for Datasets")

        # Read the HDF5 file
        with h5py.File(hdf5_file_path, 'r') as hdf5_file:
            # Get a list of all dataset names (keys)
            dataset_keys = list(hdf5_file.keys())

            # Initialize an empty DataFrame to store L2 norm distances
            # df_distances = pd.DataFrame(index=dataset_keys, columns=dataset_keys)
            df_distances = np.zeros((len(dataset_keys), len(dataset_keys)))


            # Calculate L2 norm distances between arrays
            for key1 in tqdm(dataset_keys):
                for key2 in dataset_keys:
                    # Access array data

                    array1 = hdf5_file[key1][()]
                    array2 = hdf5_file[key2][()]
                    if key1 == key2:
                        array2 = find_orthogonal_vector_numpy(array2)

                    # Calculate L2 norm distance
                    # distance = np.linalg.norm(min_max_scale(array1) - min_max_scale(array2))

                    # Calculate Cosine Sim distance
                    # distance = cosine_similarity(array1, array2)[0, 0]

                    # Calculate SinkHorn distance
                    distance = sinkhorn_pointcloud_pytorch(torch.tensor(array1, device="cuda"),
                                                torch.tensor(array2, device="cuda"),
                                                )
                    distance = distance[0].item()

                    #last idx will be distance based entire data for reference purpose
                    i = int(key1) if key1 != "all" else len(dataset_keys)-1
                    j = int(key2) if key2 != "all" else len(dataset_keys)-1
                    # Store the distance in the DataFrame
                    df_distances[i, j] = distance

        return df_distances





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

    def accum_one_sample(self,x, _):
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

    def accum_one_sample(self,x, _):
        x = x.flatten()
        self.csobj.accumulateVec(x)
        self.counter+=1

    def get_full_aggregate(self):
        res = self.csobj.table
        return res.detach().cpu().numpy()


class Feature_CountSketching():

    def __init__(self, model_name, weight_path=None, device="cuda") -> None:

        self.model = getModel(model_name)
        self.model = self.model.to(device)
        self.model.fc = torch.nn.Identity()

        if weight_path: self.model = pretrained_weight_loader(self.model, weight_path)

        self.model.eval()
        self.csobj = CountSketchVec(d=512, c=32, r=16, device=device)
        self.counter = 0

    def accum_one_sample(self,x, _):
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
        # "feature-only": ModelFeatureMean(model_name=cfg.model,
        #                                 weight_path=cfg.weight_path,
        #                                 device="cuda"),
        # "feature-Covar": ModelFeatureCovariance(model_name=cfg.model,
        #                                         weight_path=cfg.weight_path,
        #                                         device="cuda"),

        "feature-Gauss": ModelFeatureGaussDistribution( model_name=cfg.model,
                                                    weight_path=cfg.weight_path,
                                                    classwise=False, device="cuda"),
        "feature-GaussMixture": ModelFeatureGaussDistribution(model_name=cfg.model,
                                                    total_points=cfg.cloud_point_count,
                                                    weight_path=cfg.weight_path,
                                                    classwise=True, device="cuda"),

        # "feature-CountSketch": Feature_CountSketching(model_name=cfg.model,
        #                                               weight_path=cfg.weight_path,
        #                                               device="cuda"),
        # "image-Histogram" : Image_Histogram(img_size,  channels=channels,device="cuda"),
        # "image-CountSketch": Image_CountSketching(img_size, channels=channels,device="cuda"),

    }
    return sketcher_dict[cfg.sketch_method]





def data_sketch_main(cfg, file_suffix=""):

    rutl.START_SEED()
    gpu_device = torch.device("cuda")
    torch.cuda.device(gpu_device)
    print("GPU device", gpu_device)

    sfolderpath = cfg.checkpoint_dir+"/"+ cfg.sketch_method +"/"
    os.makedirs(sfolderpath, exist_ok=True)

    h5path = f"{sfolderpath}/{cfg.dataset}{cfg.dataset_type}-C{cfg.data_centers_count}-{file_suffix}.h5"
    h5file = h5py.File(h5path,"w")


    ### Centerwise data distribution / sketch
    center_list = ["all"] + list(range(cfg.data_centers_count))

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

            # all center in below cases won't be in the iided split  >>>
            if (CFG.dataset in ["CIFAR", "HUMBLE"]) and (center_index == "all"):
                last_dir = list(filter(None, cfg.weight_root_path.split("/")))[-1]
                cfg.weight_path = f"{cfg.weight_root_path.replace(last_dir, '')}/center_{center_index}/weights/bestmodel.pth"

        sketcher = getSketcher(cfg)

        with torch.no_grad():
            for laoder in laoder_list:
                for img, c in tqdm(laoder):
                    img = img.to(gpu_device, non_blocking=True)
                    sketcher.accum_one_sample(img, c)

        h5file.create_dataset(str(center_index), data=sketcher.get_full_aggregate(),
                              dtype=float)
        del sketcher
    h5file.close()

    ### Compute Distance Matrix
    gsketcher = getSketcher(cfg)
    dist_matrix = gsketcher.compute_distance_matrix(h5path)

    with h5py.File(h5path, 'a') as hdf5_file:
        hdf5_file.create_dataset("dist_matrix",
                                data=dist_matrix,
                                dtype=float)


    return h5path


### ============================================================================



def run_for_cifar(type_=10):
    CFG.dataset   = "CIFAR"

    if type_ == 100:
        weight_root_path = "/home/joseph.benjamin/WERK/fed-cvpr/fed-sketch/hypotheses/Cls1-cifar/Ex00-Cls-base_IID-002/"
        CFG.data_path = "/home/joseph.benjamin/WERK/fed-cvpr/data/cifar100-jfed/"
        CFG.csv_name = "cifar100_train_data.csv"
        CFG.data_centers_count = 10
        CFG.image_size = 224
        CFG.dataset_type = "cifar100"
        CFG.label_type = "fine_label"

    if type_ == 10:
        weight_root_path = "/home/joseph.benjamin/WERK/fed-cvpr/fed-sketch/hypotheses/Cls2-cifar10/Ex00-Cls-base_IID-001/"
        CFG.data_path = "/home/joseph.benjamin/WERK/fed-cvpr/data/cifar10-jfed/"
        CFG.csv_name = "cifar10_train_data.csv"
        CFG.data_centers_count = 5
        CFG.image_size = 224
        CFG.dataset_type = "cifar10"
        CFG.label_type = "label"

    # for alp in [100, 10, 1, 0]:  # 1000, 0.5]
    #         CFG.dirichlet_alpha = alp
    #         if weight_root_path:
    #             CFG.weight_root_path = f"{weight_root_path}/{alp}_aleph/"
    #         data_sketch_main(CFG, file_suffix=f"{alp}-")

    for alp in ["non", "full", "semi-mix", "semi-pure"]:
            if weight_root_path:
                CFG.weight_root_path = f"{weight_root_path}/{alp}_iid/"
            CFG.iid_ness = alp
            data_sketch_main(CFG, file_suffix=f"{alp}-")


def run_for_isicflamby():
    CFG.dataset   = "ISIC"
    CFG.weight_root_path = "hypotheses/CLS1-series/Cls1-isic/E00-base(ansys/E00m-Cls-Effnetb0-002_B32_Lr5e-4/"

    CFG.data_path = "/home/joseph.benjamin/WERK/fed-cvpr/data/isic2019-jfed/"
    CFG.model = "efficientnet_b0"
    CFG.data_centers_count = 6
    CFG.image_size = 200

    data_sketch_main(CFG)


def run_for_organmnist():
    CFG.dataset   = "ORGANMNIST"
    CFG.weight_root_path = "hypotheses/Cls1-organ/E00-base(ansys/E00-ClsOr-Resnet-000_B32_Lr1e-3/"

    CFG.data_path = "/home/joseph.benjamin/WERK/fed-cvpr/data/organmnist-jfed/"
    CFG.data_centers_count = 6
    CFG.image_size = 224

    data_sketch_main(CFG)


def run_for_mnist():
    CFG.dataset      = "HUMBLE"
    CFG.dataset_type = "MNIST"
    weight_root_path = "hypotheses/Cls1-Humble/Ex00-MNIST-Cls-003/"

    CFG.data_path = "/home/joseph.benjamin/WERK/fed-cvpr/data/torch-data/"
    CFG.model = "resnet9"
    CFG.data_centers_count = 5
    CFG.image_size = 28
    CFG.channels   = 1
    CFG.disable_validation = True

    for alp in ["non", "full", "semi-mix", "semi-pure"]:
            if weight_root_path:
                CFG.weight_root_path = f"{weight_root_path}/{alp}_iid/"
            CFG.iid_ness = alp
            data_sketch_main(CFG, file_suffix=f"{alp}-")



if __name__ == '__main__':
    CFG.dataset_type = "" ##for default

    # CFG.sketch_method = "feature-only"
    # CFG.sketch_method = "feature-Covar"

    # CFG.sketch_method = "feature-Gauss"
    CFG.sketch_method = "feature-GaussMixture"

    # CFG.sketch_method = "feature-CountSketch"
    # CFG.sketch_method = "image-CountSketch"
    # CFG.sketch_method = "image-Histogram"
    # CFG.sketch_method = "imagepix-RaceSketch"

    # run_for_mnist()
    run_for_cifar(10)
    # run_for_isicflamby()
    # run_for_organmnist()



