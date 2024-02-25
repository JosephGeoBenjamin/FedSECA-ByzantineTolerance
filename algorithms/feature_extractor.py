import os
import torch
import torchvision
from torch import nn

from torchvision.models.resnet import BasicBlock, ResNet

def freeze_weights(model):
    print("Freezing Resnet weights ...")
    for param in model.parameters():
        param.requires_grad = False

    return model


def load_BasicBackbone(arch, torch_pretrain= None, freeze= False):
    if torch_pretrain in ["DEFAULT", "IMAGENET-1K"]:
        torch_pretrain = "DEFAULT"
    elif torch_pretrain in [None, "NONE", "none", "None"]:
        torch_pretrain = None
    else:
        raise ValueError("Unknown pretrain weight type requested ", torch_pretrain )
    print("Torch Pretrain Set to ...", torch_pretrain)

    if arch == 'basic-alexnet':
        backbone = torchvision.models.alexnet(weights=torch_pretrain)
        outfeat_size = 256
        backbone.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        backbone.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported Model Implementation {arch} called in {os.path.basename(__file__)}")

    if freeze: backbone = freeze_weights(backbone)

    return backbone, outfeat_size




def resnet9(**kwargs):
    return ResNet(BasicBlock, [1,1,1,1],**kwargs)


def load_ResnetBackbone(arch, torch_pretrain= None, freeze= False):

    norm_layer = None
    ## TODO: add below for custom norm support
    # sd = models.resnet18(pretrained=True).state_dict()
    # model.load_state_dict(sd, strict=False)

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
                            weights=torch_pretrain, norm_layer = norm_layer)
        outfeat_size = 512

    elif arch == 'resnet50':
        backbone = torchvision.models.resnet50(zero_init_residual=True,
                            weights=torch_pretrain, norm_layer= norm_layer)
        outfeat_size = 2048

    elif arch == 'resnet9':
        backbone  = resnet9(norm_layer= norm_layer)
        outfeat_size = 512

    else:
        raise ValueError(f"Unsupported Model Implementation {arch} called in {os.path.basename(__file__)}")
    backbone.fc = nn.Identity() #remove fc of default arch

    # freeze model
    if freeze: backbone = freeze_weights(backbone)

    return backbone, outfeat_size


def load_EfficientnetBackbone(arch, torch_pretrain= None, freeze= False):

    ## pretrain setting
    if torch_pretrain in ["DEFAULT", "IMAGENET-1K"]:
        torch_pretrain = "DEFAULT"
    elif torch_pretrain in [None, "NONE", "none", "None"]:
        torch_pretrain = None
    else:
        raise ValueError("Unknown pretrain weight type requested ", torch_pretrain )
    print("Torch Pretrain Set to ...", torch_pretrain)

    ## Model loading
    if arch == 'efficientnet_b0':
        backbone = torchvision.models.efficientnet_b0(zero_init_residual=True,
                                weights=torch_pretrain)
        outfeat_size = 1280
    else:
        raise ValueError(f"Unsupported Model Implementation {arch} called in {os.path.basename(__file__)}")
    backbone.classifier = nn.Identity()  #remove fc of default arch

    # freeze model
    if freeze: backbone = freeze_weights(backbone)

    return backbone, outfeat_size


def getBackboneNetwork(arch, torch_pretrain= None, freeze= False):

    if "resnet" in arch:
        network = load_ResnetBackbone(arch, torch_pretrain, freeze)
    elif "efficientnet" in arch:
        network = load_EfficientnetBackbone(arch, torch_pretrain, freeze)
    elif "basic" in arch:
        network = load_BasicBackbone(arch, torch_pretrain, freeze)
    else:
        raise ValueError("Unknown Model Arch specified in get")

    print("Loaded Network:", arch)
    return network
