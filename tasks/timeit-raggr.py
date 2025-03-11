import os, sys
import time, datetime
import argparse, json
import copy
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as torch_F
import torchinfo
from tqdm.autonotebook import tqdm

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl
import utilities.fedUtils as fedutl

import algorithms.federation_ops as fedops
import algorithms.federation_byz as fedbyz
from algorithms.classifier import ClassifierNet


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")

##============================= Configure and Setup ============================
CFG = rutl.ObjDict(
set_device = torch.device("cpu"), #DEVICE
data_centers_count = 8,  #CLIENTS
# timeit_rounds = 100,
timeit_rounds = 1,

featx_arch     = "resnet18",  #MODEL
featx_pretrain = "NONE" , # "IMAGENET-1K" or None``
featx_dropout  = 0.0,
featx_freeze   = False,
featx_bnorm    = False,

clsfy_layers   = [9], #First mlp inwill be set w.r.t FeatureExtractor
clsfy_dropout  = 0.0,
byztn_cfg      = {},


print_freq_lstep   = 0,
ckpt_freq_Gstep    = 1,
test_last_E_epochs = 5,  # detailed cross-client cross-data testing
test_trend_full    = False, # test with pooled test for all epochs

checkpoint_dir= "/l/users/ibrahim.almakky/joseph/wacv25/clTimeComplex/run2/",
resume_training = False
)

CFG.gLogPath = CFG.checkpoint_dir
if os.path.exists(CFG.gLogPath) and (not CFG.resume_training):
    raise Exception("Logging folder already exists SO Somethings Wrong!",
                    CFG.checkpoint_dir)
if not os.path.exists(CFG.gLogPath): os.makedirs(CFG.gLogPath)


##==============================================================================
##------------------------------------------------------------------------------

def empty_create_model(cdevice=CFG.set_device):
    return ClassifierNet(arch=CFG.featx_arch,
                    fc_layer_sizes    = CFG.clsfy_layers,
                    feature_dropout   = CFG.featx_dropout,
                    classifier_dropout= CFG.clsfy_dropout,
                    feature_freeze    = CFG.featx_freeze,
                    feature_bnorm     = CFG.featx_bnorm,
                    ).to(cdevice)

##------------------------------------------------------------------------------
### Fill memory -- check to stop accidental GPU use
# DEVICES=[0]
# size_gb = 31
# size_bytes = size_gb * 1024**3
# num_elements = size_bytes // 4  # float32 takes 4 bytes
# for x in DEVICES:
#     tensor = torch.ones(num_elements, device=f"cuda:{x}")
# print("Allocation Complete...")

##------------------------------------------------------------------------------
all_defns_json = json.load(open("/home/ibrahim.almakky/joseph/wacv25/main-frame/configs/Byzantine-available-methods.json",
                            'rt'))["DEFENSE_METHODS"]


for dcc in [4, 8, 16, 32, 64, 128]:
    CFG.data_centers_count = dcc

    ## ---- Create Empty Weights
    lsets = []
    for di in range(CFG.data_centers_count):
        lset = {}
        out_state = empty_create_model().state_dict()
        lset["model_state"] = out_state
        lsets.append(lset)

    ## ---- Run Defenses
    for defns_json in all_defns_json:
        model = empty_create_model()
        CFG["defense_cfg"] = defns_json

        # fedProtocol = fedbyz.NoGuardΞByzantine  #default
        fedProtocol = getattr(fedbyz, defns_json["defense_method"])
        global_fedprtcl = fedProtocol(CFG, "G", model, CFG.set_device)

        times_log = []
        for i in tqdm(range(CFG.timeit_rounds)):
            start_time = time.time()
            global_aggset = global_fedprtcl.aggregate_globally(lsets, device=CFG.set_device)
            end_time = time.time()
            times_log.append(end_time-start_time)

        times_arr =  np.asarray(times_log)
        lutl.LOG2DICTXT({"DEF":       defns_json['defense_method'],
                        "CCOUNT": len(lsets),
                        "MEAN":   times_arr.mean(),
                        "STD":   times_arr.std(), },
                        CFG.gLogPath +'/timing_computed.txt'
                        )
    ## ----
