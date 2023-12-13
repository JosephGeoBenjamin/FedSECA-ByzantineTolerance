import os
import torch
import torchvision
from torch import nn


def freeze_weights(model):
    print("Freezing Resnet weights ...")
    for param in model.parameters():
        param.requires_grad = False

    return model


class MnistNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



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
    elif arch == 'basic-mnistnet':
        backbone = MnistNet()
        outfeat_size = 256
        backbone.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported Model Implementation {arch} called in {os.path.basename(__file__)}")

    if freeze: backbone = freeze_weights(backbone)

    return backbone, outfeat_size


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
        raise ValueError(f"Unsupported Model Implementation {arch} called in {os.path.basename(__file__)}")
    backbone.fc = nn.Identity() #remove fc of default arch

    # freeze model
    if freeze: backbone = freeze_weights(backbone)

    return backbone, outfeat_size



def getBackboneNetwork(arch, torch_pretrain= None, freeze= False):

    if "resnet" in arch:
        network = load_ResnetBackbone(arch, torch_pretrain, freeze)
    if "basic" in arch:
        network = load_BasicBackbone(arch, torch_pretrain, freeze)
    else:
        raise ValueError("Unknown Model Arch specified in get")

    print("Loaded Network:", arch)
    return network
