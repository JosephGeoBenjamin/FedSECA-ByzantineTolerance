import numpy as np
import random
from PIL import ImageOps, ImageFilter


import torchvision.transforms as torch_transforms
from torchvision.transforms import InterpolationMode

import cv2
cv2.setNumThreads(1) #Fix for Albu threads
cv2.ocl.setUseOpenCL(False)
import albumentations as Albu
import albumentations.pytorch as Albu_torch

## ==============================================================================

CIFAR_MEAN_STD    = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

## ===================== ISIC related ============================================

class IsicClassifyAuguments:
    """ NOTE:  W.R.T albumentation spawning multiple threads than set
    Use cv2 thread setting as temporary fix
    https://github.com/albumentations-team/albumentations/issues/1246
    """
    def __init__(self, image_size=200, method="kaggle"):
        self.image_size =  image_size
        print("IMAGESIZE SET::", self.image_size)

        ## Reference Kaggle SIIM-ISIC 2020 first place solution
        ## https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/175412
        transform_kaggle = Albu.Compose([
                Albu.Transpose(p=0.5),
                Albu.VerticalFlip(p=0.5),
                Albu.HorizontalFlip(p=0.5),
                Albu.RandomBrightnessContrast(0.2, 0.2,p=0.75),
                Albu.OneOf([
                    Albu.MotionBlur(blur_limit=(3,5)),
                    Albu.MedianBlur(blur_limit=(3,5)),
                    Albu.GaussianBlur(blur_limit=(3,5)),
                    Albu.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.7),

                Albu.OneOf([
                    Albu.OpticalDistortion(distort_limit=1.0),
                    Albu.GridDistortion(num_steps=5, distort_limit=1.),
                    Albu.ElasticTransform(alpha=3),
                ], p=0.7),

                Albu.CLAHE(clip_limit=4.0, p=0.7),
                Albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                    val_shift_limit=10, p=0.5),
                Albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                                    rotate_limit=15, border_mode=0, p=0.85),
                Albu.Resize(self.image_size, self.image_size),
                Albu.CoarseDropout(random.randint(1, 8), 16, 16, p=0.7), # modified to flamby
                Albu.Normalize(),
                Albu_torch.ToTensorV2(),
        ])

        transform_flamby = Albu.Compose([
                Albu.RandomScale(0.07),
                Albu.Rotate(50),
                Albu.RandomBrightnessContrast(0.15, 0.1),
                Albu.Flip(p=0.5),
                Albu.Affine(shear=0.1),
                Albu.RandomCrop(self.image_size, self.image_size),
                Albu.CoarseDropout(random.randint(1, 8), 16, 16),
                Albu.Normalize(always_apply=True),
                Albu_torch.ToTensorV2(),

        ])

        transform_infer = Albu.Compose([
                Albu.CenterCrop(self.image_size, self.image_size),
                Albu.Normalize(always_apply=True),
                Albu_torch.ToTensorV2(),
                ])

        if   method == "kaggle":
            self.transform_main = transform_kaggle
        elif method == "flamby":
            self.transform_main = transform_flamby
        elif method == "infer":
            self.transform_main = transform_infer
        else :
            raise ValueError("Unknown Augmentation Type for ISIC transforms")


    def __call__(self, x):
        x_a = np.asarray(x)
        y2 = self.transform_main(image=x_a)['image']
        return y2

    def get_composition(self):
        return str(self.transform_main)


## ===================== Cifar100 Transforms ==============================

class CifarClassifyAuguments:
    def __init__(self, image_size=224, method="train"):
        self.image_size =  image_size
        print("IMAGESIZE SET::", self.image_size)

        data_mean = IMAGENET_MEAN_STD[0]
        data_std  = IMAGENET_MEAN_STD[1]

        transform_train = torch_transforms.Compose([
            torch_transforms.Resize((self.image_size, self.image_size),
                                    interpolation=InterpolationMode.BICUBIC),
            torch_transforms.RandomHorizontalFlip(),
            torch_transforms.RandomRotation(10),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=data_mean, std=data_std)
            ])

        transform_infer = torch_transforms.Compose([
            torch_transforms.Resize((self.image_size,self.image_size)),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=data_mean, std=data_std)
            ])

        if   method == "train":
            self.transform_main = transform_train
        elif method == "infer":
            self.transform_main = transform_infer
        else :
            raise ValueError(f"Unknown Augmentation Type for Cifar100 {method} transforms")


    def __call__(self, x):
        y2 = self.transform_main(x)
        return y2

    def get_composition(self):
        return str(self.transform_main)

## ===================== OrganMNIST Transforms ==============================


class OrganMnistClassifyAuguments:
    ## Note: also refer to Kaggle RSNA-STR Pulmonary Embolism solutions
    ## https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/discussion/194145

    def __init__(self, method = "train", image_size = 224):
        self.image_size =  image_size
        print("IMAGESIZE SET::", self.image_size)

        data_mean = 0.5 #IMAGENET_MEAN_STD[0]
        data_std  = 0.5 #IMAGENET_MEAN_STD[1]

        train_transform = torch_transforms.Compose([
            torch_transforms.Resize(image_size,
                                    interpolation=InterpolationMode.BICUBIC),
            # torch_transforms.RandAugment(num_ops=3, magnitude=5),
            torch_transforms.RandomHorizontalFlip(p=0.5),
            torch_transforms.RandomVerticalFlip(p=0.5),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=data_mean, std=data_std),
            # torch_transforms.RandomErasing(p=0.5, value=0),
        ])

        infer_transform = torch_transforms.Compose([
            torch_transforms.Resize(image_size,
                        interpolation=InterpolationMode.BICUBIC),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=data_mean, std=data_std)
        ])

        if   method == "train":
            self.transform_main = train_transform
        elif method == "infer":
            self.transform_main = infer_transform
        else : raise ValueError("Unknown Mode set only `train` or `infer` allowed")

    def __call__(self, x):
        y = self.transform_main(x)
        return y

    def get_composition(self):
        return str(self.transform_main)


##====================== Humble (MNIST) Transforms =============================



class HumbleAuguments:
    def __init__(self, C1_to_C3 = True):
        print("Humble Transforms",)

        transforms_list = []
        if C1_to_C3:
            transforms_list.append(torch_transforms.transforms.Grayscale(num_output_channels=3))
        transforms_list.append(torch_transforms.transforms.ToTensor(),)
        transforms_list.append(torch_transforms.transforms.Normalize((0.5,), (0.5,)))

        self.transform_main = torch_transforms.Compose(transforms_list)


    def __call__(self, x):
        y = self.transform_main(x)
        return y

    def get_composition(self):
        return str(self.transform_main)


##====================== iNaturalist Transforms =============================

class DeiTAuguments:
    ## Note: Adapted from DeiT https://github.com/facebookresearch/deit/blob/main/augment.py
    ## Data-augmentation (DA) based on dino DA (https://github.com/facebookresearch/dino
    ##    and timm DA(https://github.com/rwightman/pytorch-image-models)
    ## Used for iNaturalist in my repo

    def __init__(self, method = "train", image_size = 224):
        self.image_size =  image_size
        print("IMAGESIZE SET::", self.image_size)

        data_mean = IMAGENET_MEAN_STD[0]
        data_std  = IMAGENET_MEAN_STD[1]

        ##--------
        primary_tfl = [
            torch_transforms.Resize(image_size, interpolation=3),
            torch_transforms.RandomCrop(image_size, padding=4,padding_mode='reflect'),
            torch_transforms.RandomHorizontalFlip()
        ]
        secondary_tfl = [torch_transforms.RandomChoice([
            torch_transforms.Grayscale(num_output_channels=3),
            torch_transforms.RandomSolarize(threshold=128, p=1.0),
            torch_transforms.GaussianBlur(kernel_size=(9), sigma=(0.1, 2.0)),
            torch_transforms.ColorJitter(0.3)
            ])]
        ##--------

        train_transform = torch_transforms.Compose(
            primary_tfl+secondary_tfl+[
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=data_mean, std=data_std),
            # torch_transforms.RandomErasing(p=0.5, value=0),
        ])

        infer_transform = torch_transforms.Compose([
            torch_transforms.Resize(image_size,
                        interpolation=InterpolationMode.BICUBIC),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=data_mean, std=data_std)
        ])

        if   method == "train":
            self.transform_main = train_transform
        elif method == "infer":
            self.transform_main = infer_transform
        else : raise ValueError("Unknown Mode set only `train` or `infer` allowed")

    def __call__(self, x):
        y = self.transform_main(x)
        return y

    def get_composition(self):
        return str(self.transform_main)
