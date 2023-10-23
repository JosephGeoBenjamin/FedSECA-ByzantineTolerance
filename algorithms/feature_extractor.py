import os
import torch
import torchvision
from torch import nn


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
    elif arch == 'resnet34':
        backbone = torchvision.models.resnet34(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 512
    elif arch == 'resnet50':
        backbone = torchvision.models.resnet50(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 2048

    elif arch == 'resnet101':
        backbone = torchvision.models.resnet101(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 2048

    elif arch == 'resnet152':
        backbone = torchvision.models.resnet152(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 2048

    else:
        raise ValueError(f"Unknown Model Implementation called in {os.path.basename(__file__)}")
    backbone.fc = nn.Identity() #remove fc of default arch

    # freeze model
    if freeze:
        print("Freezing Resnet weights ...")
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, outfeat_size


def load_EfficientnetBackbone(arch, torch_pretrain= None, freeze= False):

    ## pretrain setting
    if torch_pretrain in ["DEFAULT", "IMAGENET-1K"]:
        torch_pretrain = "IMAGENET1K_V1"
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
    elif arch == 'efficientnet_b1':
        backbone = torchvision.models.efficientnet_b1(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 1280
    elif arch == 'efficientnet_b2':
        backbone = torchvision.models.efficientnet_b2(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 1408
    elif arch == 'efficientnet_b3':
        backbone = torchvision.models.efficientnet_b3(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 1536
    elif arch == 'efficientnet_b4':
        backbone = torchvision.models.efficientnet_b4(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 1792
    elif arch == 'efficientnet_b5':
        backbone = torchvision.models.efficientnet_b5(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 2048
    elif arch == 'efficientnet_b6':
        backbone = torchvision.models.efficientnet_b6(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 2304
    elif arch == 'efficientnet_b7':
        backbone = torchvision.models.efficientnet_b7(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 2560


    elif arch == 'efficientnet_v2_s':
        backbone = torchvision.models.efficientnet_v2_s(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 1280
    elif arch == 'efficientnet_v2_m':
        backbone = torchvision.models.efficientnet_v2_m(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 1280
    elif arch == 'efficientnet_v2_l':
        backbone = torchvision.models.efficientnet_v2_l(zero_init_residual=True,
                            weights=torch_pretrain)
        outfeat_size = 1280


    else:
        raise ValueError(f"Unknown Model Implementation called in {os.path.basename(__file__)}")
    backbone.classifier = nn.Identity() #remove fc of default arch

    # freeze model
    if freeze:
        print("Freezing EfficientNet weights ...")
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, outfeat_size

# resnet.__dict__[args.arch](zero_init_residual=True)



def getBackboneNetwork(arch, torch_pretrain= None, freeze= False):

    if "resnet" in arch:
        network = load_ResnetBackbone(arch, torch_pretrain, freeze)
    elif "efficientnet" in arch:
        network = load_EfficientnetBackbone(arch, torch_pretrain, freeze)
    else:
        raise ValueError("Unknown Model Arch specified in get")

    print("Loaded Network:", arch)
    return network
