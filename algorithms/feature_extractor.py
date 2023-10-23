import os
import torch
import torchvision
from torch import nn


def freeze_weights(model):
    print("Freezing Resnet weights ...")
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_ResnetBackbone(arch, torch_pretrain= None, freeze= False):

    ## pretrain setting
    if torch_pretrain in ["DEFAULT", "IMAGENET-1K"]:
        torch_pretrain = "DEFAULT"
    elif torch_pretrain in [None, "NONE", "none", "None"]:
        torch_pretrain = None
    else:
        raise ValueError("Unknown pretrain weight type requested ", torch_pretrain )
    print("Torch Pretrain Set to ...", torch_pretrain)

    ## Model loading
    if arch == 'resnet18':
        backbone = torchvision.models.resnet18(zero_init_residual=True,
                                weights=torch_pretrain)
        outfeat_size = 512

    elif arch == 'resnet50':
        backbone = torchvision.models.resnet50(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 2048

    else:
        raise ValueError(f"Unsupported Model Implementation called in {os.path.basename(__file__)}")
    backbone.fc = nn.Identity() #remove fc of default arch

    # freeze model
    if freeze: backbone = freeze_weights(backbone)

    return backbone, outfeat_size



def getBackboneNetwork(arch, torch_pretrain= None, freeze= False):

    if "resnet" in arch:
        network = load_ResnetBackbone(arch, torch_pretrain, freeze)
    else:
        raise ValueError("Unknown Model Arch specified in get")

    print("Loaded Network:", arch)
    return network
