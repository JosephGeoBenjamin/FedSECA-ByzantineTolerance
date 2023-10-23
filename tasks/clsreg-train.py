import os, sys
import time, datetime
import argparse, json

import numpy as np
import torch
from torch import nn, optim

import torchinfo
from tqdm.autonotebook import tqdm

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl
from utilities.metricUtils import MultiClassMetrics

from algorithms.vne import compute_VNEntropy
from algorithms.vicreg import compute_AutoVarCovLoss
from algorithms.barlow  import compute_AutoCorrBTLoss
from algorithms.classifier import ClassifierNet, ClassifierWithProjectorNet



print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")

##============================= Configure and Setup ============================

CFG = rutl.ObjDict(
dataset = "ISIC",
data_root_path  = "/home/joseph.benjamin/WERK/fed-cvpr/data/isic2019-jfed",
data_centers_count = 6,
test_partitions = 6,
override_csv = None,

## data specifics
dirichlet_alpha = None, # for Cifar100

epochs        = 100,
image_size    = 200,
batch_size    = 64,
workers       = 8,

learning_rate = 1e-4,
weight_decay  = 1e-6,
enable_scheduler = True,

reg_method    = "VIC", ## VNE ; VIC ; BT
reg_coeff     = 1,  ## vne:0.5 ; vic:0.05 bt:0.001
projector    = [ 2048, 2048, 2048 ],

featx_arch     = "resnet18",
featx_pretrain = "IMAGENET-1K" , # "IMAGENET-1K" or None
featx_dropout  = 0.2,
featx_freeze   = False,
featx_bnorm    = False,

clsfy_layers   = [9], #First mlp inwill be set w.r.t FeatureExtractor
clsfy_dropout  = 0.5,

checkpoint_dir= "hypotheses/#dummy-run/trail-001",
resume_training=False
)

### -----
parser = argparse.ArgumentParser(description='Classification task')
parser.add_argument('--load-json', type=str, metavar='JSON',
    help='Load settings from file in json format which override values hard codes in py file.')

parser.add_argument('--featx-pretrain', type=str, metavar='PATH',
    help='Set from where to load the prestrained weight from')

parser.add_argument('--checkpoint-dir', type=str, metavar='PATH',
    help='Load settings from file in json format. Command line options override values in file.')


args = parser.parse_args()

if args.load_json:
    with open(args.load_json, 'rt') as f:
        CFG.__dict__.update(json.load(f))

for arg in vars(args):
    att = getattr(args, arg)
    if att: CFG.__dict__[arg] = att


### ----------------------------------------------------------------------------
CFG.gLogPath = CFG.checkpoint_dir
CFG.gWeightPath = CFG.gLogPath+"/weights/"
## Perform config checks

def save_current_configs(cfg):
    with open(cfg.gLogPath+"/exp_config.json", "a") as f:
        json.dump(vars(cfg), f, indent=4)
    #save this  main file for debugging
    with open(os.path.realpath(__file__), "r") as pymain:
        file_contents = pymain.read()
        file_contents = str(datetime.datetime.now()) +"\n"+file_contents
    with open(cfg.gLogPath+"/python-main.txt", "a") as save_text:
        save_text.write(file_contents)

### ============================================================================

def getDataLoaders(cfg, center_index, type="train"):

    if cfg.dataset == "ISIC":
        from datacode.isic_jfed_data import getIsicCLSLoaders as trainloader
        from datacode.isic_jfed_data import getIsicTESTLoader as testloader
    elif cfg.dataset == "CIFAR":
        from datacode.cifar100_jfed_data import getCifar100CLSLoaders as trainloader
        from datacode.cifar100_jfed_data import getCifar100TESTLoader as testloader
    elif cfg.dataset == "ORGAN-MNIST":
        from datacode.orgmnist_jfed_data import getOrganMnistCLSLoaders as trainloader
        from datacode.orgmnist_jfed_data import getOrganMnistTESTLoader as testloader
    else:
        raise ValueError(f"Unsupported data type specfied {cfg.dataset}")

    if type =="train":
        return trainloader(cfg=cfg, center_index = center_index)
    elif type =="test":
        return testloader(cfg=cfg, center_index = center_index)
    else:
        raise ValueError("train/test Type unknown")

    return None

### ============================================================================


def getModelnOptimizer(model_key=None, device="cpu"):

    ## pretrain setting
    m_state = 0; torch_pretrain_flag = None
    if os.path.isfile(CFG.featx_pretrain):
        m_state = torch.load(CFG.featx_pretrain, map_location='cpu')
    else: torch_pretrain_flag = CFG.featx_pretrain

    model = ClassifierWithProjectorNet(arch=CFG.featx_arch,
                    fc_layer_sizes    = CFG.clsfy_layers,
                    feature_dropout   = CFG.featx_dropout,
                    classifier_dropout= CFG.clsfy_dropout,
                    feature_freeze    = CFG.featx_freeze,
                    feature_bnorm     = CFG.featx_bnorm,
                    torch_pretrain    = torch_pretrain_flag,
                    projector_sizes   = CFG.projector
                    )

    if m_state:
        key = model_key if model_key else "model"
        ret_msg = model.load_state_dict(m_state[key], strict=False)
        lutl.LOG2TXT(f"Manual Pretrain Loaded...{CFG.featx_pretrain},{str(ret_msg)}",
                     CFG.gLogPath +'/misc.txt')

    model_info = torchinfo.summary(model, (1, 3, CFG.image_size, CFG.image_size),
                                verbose=0)
    lutl.LOG2TXT(model_info, CFG.gLogPath +'/misc.txt', console= False)

    model.to(device)

    ##-------------- OPTIM ------------------------

    optimizer = optim.AdamW(model.parameters(), lr=CFG.learning_rate,
                        weight_decay=CFG.weight_decay)

    scheduler_list = False
    scheduler_list = []
    if CFG.enable_scheduler:
        if CFG.dataset == "ORGAN-MNIST":
            scheduler_list.append(optim.lr_scheduler.MultiStepLR( optimizer,
                            milestones=[50, 75], gamma=0.1))
        elif CFG.dataset == "ISIC":
            scheduler_list.append(optim.lr_scheduler.MultiStepLR( optimizer,
                                    milestones=[50, 75], gamma=0.1))
        elif CFG.dataset == "CIFAR":
            scheduler_list.append(optim.lr_scheduler.MultiStepLR( optimizer,
                milestones=[50, 75], gamma=0.1))
        else: raise(f"Scheduler Enabled but Dataset class unknown {CFG.dataset}")

        lutl.LOG2TXT(f"Schedulers Enabled: {scheduler_list}",
                    CFG.gLogPath +'/misc.txt', console= False)
    else: scheduler_list= False

    return model, optimizer, scheduler_list


### ============================================================================


def getLossFunc():
    # lossfn = WeightedFocalLoss(alpha=class_weights, gamma=0.2)
    ce_loss = nn.CrossEntropyLoss()

    if   CFG.reg_method == "VNE": reg_loss = compute_VNEntropy
    elif CFG.reg_method == "VIC": reg_loss = compute_AutoVarCovLoss
    elif CFG.reg_method == "BT":  reg_loss = compute_AutoCorrBTLoss
    else: reg_loss = lambda x: torch.tensor(0)

    def lossfunc(pred, tgt, feature):
        ce  = ce_loss(pred, tgt)
        rl  = reg_loss(feature)
        return  (ce+ CFG.reg_coeff*rl) ,  {"CE":ce.item(), "REG":rl.item()}

    lutl.LOG2TXT(f"Using the Loss::::{CFG.reg_method}",  CFG.gLogPath +'/misc.txt')
    return lossfunc


### ============================================================================


def simple_main(model_key=None, center_index=None, folder_suffix=""):


    ### SETUP
    rutl.START_SEED()
    gpu_device = torch.device("cuda")
    torch.cuda.device(gpu_device)
    print("GPU device", gpu_device)

    # -- log path --
    if model_key: folder_suffix +=f"/{model_key}/"
    if not (center_index==None): folder_suffix +=f"/center_{center_index}/"


    CFG.gLogPath = CFG.checkpoint_dir+folder_suffix
    CFG.gWeightPath = CFG.gLogPath+"/weights/"

    if os.path.exists(CFG.gLogPath) and (not CFG.resume_training):
        raise Exception("CheckPoint folder already exists and resume_training not enabled; Somethings Wrong!",
                        CFG.gLogPath)
    if not os.path.exists(CFG.gWeightPath): os.makedirs(CFG.gWeightPath)

    save_current_configs(CFG)

    lutl.LOG2TXT(f"C:{center_index}"*10, CFG.gLogPath +'/misc.txt', console= False)

    ### DATA ACCESS
    trainloader, validloader = getDataLoaders(CFG, center_index, type="train")

    ### MODEL, OPTIM
    model, optimizer, scheduler_list = getModelnOptimizer(model_key, gpu_device)
    lossfn = getLossFunc()

    ## Automatically resume from checkpoint if it exists and enabled
    if os.path.exists(CFG.gWeightPath +'/checkpoint.pth') and CFG.resume_training:
        ckpt = torch.load(CFG.gWeightPath  +'/checkpoint.pth',
                            map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lutl.LOG2TXT(f"Restarting Training from EPOCH:{start_epoch} of {CFG.checkpoint_dir}",  CFG.gLogPath +'/misc.txt')
    else:
        start_epoch = 0


    ### MODEL TRAINING
    start_time = time.time()
    best_acc = 0 ; best_loss = float('inf')
    trainMetric = MultiClassMetrics(CFG.gLogPath+"/metrics/", digits=4)
    validMetric = MultiClassMetrics(CFG.gLogPath+"/metrics/", digits=4)

    for epoch in range(start_epoch, CFG.epochs):

        ## ---- Training Routine ----
        model.train()
        for img, tgt in tqdm(trainloader):
            img = img.to(gpu_device, non_blocking=True)
            tgt = tgt.to(gpu_device, non_blocking=True)
            optimizer.zero_grad()
            pred, featp = model.forward(img)
            loss, loss_info = lossfn(pred, tgt, featp)
            loss.backward()
            optimizer.step()
            trainMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss, loss_info)
        if scheduler_list:
            for scheduler in scheduler_list: scheduler.step()

        ## save checkpoint states
        # state = dict(epoch=epoch + 1, model=model.state_dict(),
        #                 optimizer=optimizer.state_dict())
        # torch.save(state, CFG.gWeightPath +'/checkpoint.pth')


        ## ---- Validation Routine ----
        model.eval()
        with torch.no_grad():
            for img, tgt in tqdm(validloader):
                img = img.to(gpu_device, non_blocking=True)
                tgt = tgt.to(gpu_device, non_blocking=True)
                pred, featx = model.forward(img)
                loss, loss_info = lossfn(pred, tgt, featx)
                validMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss, loss_info)

        ## Log Metrics TODO Add balanced and F1
        stats = dict(
                epoch=epoch, time=int(time.time() - start_time),
                trainloss = trainMetric.get_loss(),
                trainacc  = trainMetric.get_balanced_accuracy(),
                trainF1   = trainMetric.get_f1score(),
                validloss = validMetric.get_loss(),
                validacc  = validMetric.get_balanced_accuracy(),
                validF1   = validMetric.get_f1score(),
                trainlossInfo = trainMetric.get_loss_info_aggregates(),
                validlossInfo = validMetric.get_loss_info_aggregates(),
                )
        lutl.LOG2DICTXT(stats, CFG.gLogPath+'/train-stats.txt')


        ## save best model
        best_flag = False
        if stats['validF1'] > best_acc:
            torch.save(model.state_dict(), CFG.gWeightPath +'/bestmodel.pth')
            best_acc = stats['validF1']
            best_loss = stats['validloss']
            best_flag = True

            ##---> Log detailed validation
            detail_stat = dict(
                    epoch=epoch, time=int(time.time() - start_time),
                    best = best_flag,
                    validbalacc = validMetric.get_balanced_accuracy(),
                    validf1scr  = validMetric.get_f1score(),
                    validreport = validMetric.get_class_report(),
                    # validconfus = validMetric.get_confusion_matrix().tolist(),
                )
            lutl.LOG2DICTXT(detail_stat, CFG.gLogPath+'/validation-details.txt', console=False)

        trainMetric.reset()
        validMetric.reset()

    return CFG.gLogPath



def simple_test(saved_logpath):

    gpu_device = torch.device("cuda")
    torch.cuda.device(gpu_device)

    ### MODEL
    model = ClassifierNet(arch=CFG.featx_arch,
                    fc_layer_sizes    = CFG.clsfy_layers,
                    feature_dropout   = CFG.featx_dropout,
                    classifier_dropout= CFG.clsfy_dropout,
                    feature_freeze    = CFG.featx_freeze,
                    feature_bnorm     = CFG.featx_bnorm,
                    )
    model = model.to(gpu_device)
    ret_msg = model.load_state_dict(torch.load(saved_logpath+"/weights/bestmodel.pth"), strict=False)
    lutl.LOG2TXT(f"Testing Weight Loaded...{CFG.featx_pretrain},{str(ret_msg)}; {saved_logpath} ",
                     CFG.gLogPath +'/misc.txt')

    ### MODEL TESTING

    test_center_num = CFG.test_partitions if CFG.test_partitions>1 else 0
    for c in ["all"]+ list(range(test_center_num)):
        testloader = getDataLoaders(CFG, center_index=c, type="test")
        testMetric = MultiClassMetrics(saved_logpath)
        model.eval()

        start_time = time.time()
        with torch.no_grad():
            for img, tgt in tqdm(testloader, disable=CFG.disable_tqdm):
                img = img.to(gpu_device, non_blocking=True)
                tgt = tgt.to(gpu_device, non_blocking=True)
                pred = model.forward(img)
                testMetric.add_entry(torch.argmax(pred, dim=1), tgt)

            ## Log detailed validation
            log_title = f"test-{c}"
            detail_stat = dict(
                    model_used  = os.path.basename(saved_logpath),
                    test_center = c,
                    timetaken   = int(time.time() - start_time),
                    testf1scr   = testMetric.get_f1score(),
                    testbalacc  = testMetric.get_balanced_accuracy(),
                    testacc     = testMetric.get_accuracy(),
                    testreport  = testMetric.get_class_report(),
                    testconfus  = testMetric.get_confusion_matrix(
                                    save_png= True, title=log_title).tolist(),
                )
            lutl.LOG2DICTXT(detail_stat, saved_logpath+'/test-results.txt',
                            console=True)

            testMetric._write_predictions(title=log_title)



if __name__ == '__main__':

    # model_list = ["global_model"]+[f"local_model_{i}" for i in range(CFG.data_centers_count)]
    # model_list = ["global_model"]
    model_list = [None]

    # center_list = ["all"] +list(range(CFG.data_centers_count))
    center_list = ["all"]

    for m in model_list:
        for c in center_list:
            logpth = simple_main(center_index=c, model_key=m)
            simple_test(logpth)
