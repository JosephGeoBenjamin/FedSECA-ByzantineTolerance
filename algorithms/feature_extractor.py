import os
import torch
import torchvision
from torch import nn
import timm
from torchvision.models.resnet import BasicBlock, ResNet

class ImageLayerNorm(nn.LayerNorm):
    def __init__(self, plane, **kwargs):
        super().__init__(plane)
        self.ln_forward = super().forward

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous() # (N, C, H, W) -> (N, H, W, C)
        x = self.ln_forward(x)
        x = x.permute(0, 3, 1, 2).contiguous() # (N, H, W, C) -> (N, C, H, W)

        return x


def freeze_weights(model):
    print("Freezing Resnet weights ...")
    for param in model.parameters():
        param.requires_grad = False

    return model

##------------------------------------------------------------------------------

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


##------------------------------------------------------------------------------


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
    if arch == 'resnet18': #11.6M params
        backbone = torchvision.models.resnet18(norm_layer = norm_layer)
        bk_state = torchvision.models.resnet18(zero_init_residual=True,
                                    weights=torch_pretrain).state_dict()
        backbone.load_state_dict(bk_state, strict=False)

        outfeat_size = 512

    elif arch == 'resnet50': #25.6M params
        backbone = torchvision.models.resnet50(norm_layer= norm_layer)
        bk_state = torchvision.models.resnet50(zero_init_residual=True,
                            weights=torch_pretrain).state_dict()
        backbone.load_state_dict(bk_state, strict=False)

        outfeat_size = 2048

    elif arch == 'resnet9': #6.5M params
        backbone  = resnet9(norm_layer= norm_layer)
        outfeat_size = 512
        if torch_pretrain: Warning(f"{arch} has no imagenet pretraining available")

    else:
        raise ValueError(f"Unsupported Model Implementation {arch} called in {os.path.basename(__file__)}")
    backbone.fc = nn.Identity() #remove fc of default arch

    # freeze model
    if freeze: backbone = freeze_weights(backbone)

    return backbone, outfeat_size


##------------------------------------------------------------------------------


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
    if arch == 'efficientnet_b0': #5.3M param
        backbone = torchvision.models.efficientnet_b0(zero_init_residual=True,
                                weights=torch_pretrain)
        outfeat_size = 1280
    else:
        raise ValueError(f"Unsupported Model Implementation {arch} called in {os.path.basename(__file__)}")
    backbone.classifier = nn.Identity()  #remove fc of default arch

    # freeze model
    if freeze: backbone = freeze_weights(backbone)

    return backbone, outfeat_size

##------------------------------------------------------------------------------

def load_ConvNextBackbone(arch, torch_pretrain= None, freeze= False):

    ## pretrain setting
    if torch_pretrain in ["DEFAULT", "IMAGENET-1K"]:
        torch_pretrain = True
    elif torch_pretrain in [None, "NONE", "none", "None"]:
        torch_pretrain = False
    else:
        raise ValueError("Unknown pretrain weight type requested ", torch_pretrain )
    print("Torch Pretrain Set to ...", torch_pretrain)

    ## Model loading

    # backbone = torchvision.models.convnext_tiny(weights=torch_pretrain) # pytorch
    if arch == 'convnext_base': #89M param
        model_card_name = "timm/convnext_base.fb_in1k"
        outfeat_size = 1024

    elif arch == 'convnext_tiny': #28.6M param
        model_card_name = "timm/convnext_tiny.fb_in1k"
        outfeat_size = 768

    elif arch == 'convnext_nano':#15.6M param
        model_card_name = "timm/convnext_nano_ols.d1h_in1k"
        outfeat_size = 640

    else:
        raise ValueError(f"Unsupported Model Implementation {arch} called in {os.path.basename(__file__)}")

    print("Loading timm Model Architecture....", model_card_name)

    backbone = timm.create_model(model_card_name,
                                pretrained=torch_pretrain,   # num_classes=0,
                                )
    backbone.head = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())

    # backbone = torchvision.models.convnext_tiny(weights=torch_pretrain)

    # freeze model
    if freeze: backbone = freeze_weights(backbone)

    return backbone, outfeat_size

##------------------------------------------------------------------------------


def load_DeitBackbone(arch, torch_pretrain= None, freeze= False):
    ## pretrain setting
    if torch_pretrain in ["DEFAULT", "IMAGENET-1K"]:
        torch_pretrain = True
    elif torch_pretrain in [None, "NONE", "none", "None"]:
        torch_pretrain = False
    else:
        raise ValueError("Unknown pretrain weight type requested ", torch_pretrain )
    print("Torch Pretrain Set to ...", torch_pretrain)

    ## Model loading
    if arch == "deit_tiny":  #5.7M param
        model_card_name = 'timm/deit_tiny_patch16_224.fb_in1k'
        outfeat_size = 192

    if arch == "deit_small": #22.1M param
        model_card_name = 'timm/deit_small_patch16_224.fb_in1k'
        outfeat_size = 384

    if arch == "deit_base":  #86.6M param
        model_card_name = 'timm/deit_base_patch16_224.fb_in1k'
        outfeat_size = 768

    else:
        raise ValueError(f"Unsupported Model Implementation {arch} called in {os.path.basename(__file__)}")
    print("Loading timm Model Architecture....", model_card_name)

    backbone = timm.create_model(model_card_name,
                                 pretrained=torch_pretrain,
                                # num_classes=0,  # remove classifier nn.Linear
                                )
    raise "Classifier Not set"
    # freeze model
    if freeze: backbone = freeze_weights(backbone)

    return backbone,


##==============================================================================

def getBackboneNetwork(arch, torch_pretrain= None, freeze= False):

    if "resnet" in arch:
        network = load_ResnetBackbone(arch, torch_pretrain, freeze)
    elif "efficientnet" in arch:
        network = load_EfficientnetBackbone(arch, torch_pretrain, freeze)
    elif "convnext" in arch:
        network = load_ConvNextBackbone(arch, torch_pretrain, freeze)
    elif "deit" in arch:
        network = load_DeitBackbone(arch, torch_pretrain, freeze)
    elif "basic" in arch:
        network = load_BasicBackbone(arch, torch_pretrain, freeze)
    else:
        raise ValueError("Unknown Model Arch specified in get")

    print("Loaded Network:", arch)
    return network
