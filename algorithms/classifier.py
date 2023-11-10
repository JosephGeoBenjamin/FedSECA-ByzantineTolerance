import os, sys
import torch
import torchvision
from torch import nn
# sys.path.append(os.getcwd())
from algorithms.feature_extractor import getBackboneNetwork


##================= CLassifier Wrapper =========================================

class ClassifierNet(nn.Module):
    def __init__(self, arch, fc_layer_sizes=[512,1000],
                    feature_dropout=0, classifier_dropout=0,
                    feature_freeze = False, feature_bnorm = False,
                    torch_pretrain=None,
                    return_gap_feature = False):
        super().__init__()

        self.fc_layer_sizes = fc_layer_sizes
        self.return_gap_feature = return_gap_feature

        ### Feature Extractor
        self.backbone,self.feat_outsize = getBackboneNetwork(arch=arch,
                                            torch_pretrain=torch_pretrain,
                                            freeze=feature_freeze)
        fx_layers = []
        if feature_bnorm:
            fx_layers.append(nn.BatchNorm1d(self.feat_outsize, affine=False))
        fx_layers.append(nn.Dropout(p=feature_dropout))

        self.featx_proc = nn.Sequential(*fx_layers)

        ### Classifier
        sizes = [self.feat_outsize] + list(self.fc_layer_sizes)
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=True))
            layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=classifier_dropout))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.backbone(x)
        out  = self.featx_proc(feat)
        out  = self.classifier(out)

        return out


##==============================================================================


if __name__ == "__main__":

    from torchinfo import summary

    model = ClassifierNet(arch='efficientnet_b0', fc_layer_sizes=[64,8],
                    feature_dropout=0, classifier_dropout=0,
                    torch_pretrain=None)
    summary(model, (1, 3, 200, 200))
    print(model)


"""
    NOTE:
    ## used for some CIFAR sized image datasets
    # self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # self.backbone.maxpool = nn.Identity()
"""