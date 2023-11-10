import os, sys
import time, datetime
import argparse, json
import copy
import numpy as np
import torch
from torch import nn, optim

import torchinfo
from tqdm.autonotebook import tqdm

sys.path.append(os.getcwd())
import utilities.runUtils as rutl
import utilities.logUtils as lutl
import utilities.fedUtils as fedutl
from utilities.metricUtils import MultiClassMetrics

from algorithms.classifier import ClassifierNet


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")

##============================= Configure and Setup ============================

CFG = rutl.ObjDict(
dataset = "ISIC",
data_root_path  = "/home/joseph.benjamin/WERK/fed-cvpr/data/isic2019-jfed",
data_centers_count = 6,
test_partitions = 6,
override_csv = None,

## Fed specifics
dirichlet_alpha = None, # for Cifar100

epochs        = 100,
image_size    = 200,
batch_size    = 64,
workers       = 2,

learning_rate   = 5e-4,
weight_decay    = 1e-6,
enable_scheduler = True,

enable_weight_reinit = True, # FedAvg protocol, true in general cases

reg_method   = "NONE",
reg_coeff    = 0.0,

featx_arch     = "resnet18",
featx_pretrain = "IMAGENET-1K" , # "IMAGENET-1K" or None``
featx_dropout  = 0.2,
featx_freeze   = False,
featx_bnorm    = False,

clsfy_layers   = [9], #First mlp inwill be set w.r.t FeatureExtractor
clsfy_dropout  = 0.5,

print_freq_lstep = 0,
ckpt_freq_Gstep  = 1,

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

def getDataLoaders(cfg, center_index=None, type="train"):

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
        traindozers = {}; validdozers = {}
        for cen in list(range(cfg.data_centers_count)):
            traindozers[cen], validdozers[cen] = trainloader(cfg=cfg, center_index = cen)
        _, validdozers["all"] = trainloader(cfg=cfg, center_index = "all")

        return traindozers, validdozers

    elif type =="test":
        return testloader(cfg=cfg, center_index = center_index)
    else:
        raise ValueError("train/test Type unknown")

    return None



def getModel(model_key=None, device="cpu"):

    ## pretrain setting
    m_state = 0; torch_pretrain_flag = None
    if os.path.isfile(CFG.featx_pretrain):
        m_state = torch.load(CFG.featx_pretrain, map_location='cpu')
    else: torch_pretrain_flag = CFG.featx_pretrain

    model = ClassifierNet(arch=CFG.featx_arch,
                    fc_layer_sizes    = CFG.clsfy_layers,
                    feature_dropout   = CFG.featx_dropout,
                    classifier_dropout= CFG.clsfy_dropout,
                    feature_freeze    = CFG.featx_freeze,
                    feature_bnorm     = CFG.featx_bnorm,
                    torch_pretrain    = torch_pretrain_flag,
                    )

    if m_state:
        key = model_key if model_key else "model"
        ret_msg = model.load_state_dict(m_state[key], strict=False)
        lutl.LOG2TXT(f"Manual Pretrain Loaded...{CFG.featx_pretrain},{str(ret_msg)}",
                     CFG.gLogPath +'/misc.txt')

    model_info = torchinfo.summary(model, (1, 3, CFG.image_size, CFG.image_size),
                                verbose=0)
    lutl.LOG2TXT(model_info, CFG.gLogPath +'/misc.txt', console= False)

    return model

### ============================================================================



def getLossFunc():
    # lossfn = WeightedFocalLoss(alpha=class_weights, gamma=0.2)
    ce_loss = nn.CrossEntropyLoss()

    if   CFG.reg_method == "TBD": reg_loss = lambda x: torch.tensor(0)
    else: reg_loss = lambda x: torch.tensor(0)

    def lossfunc(pred, tgt, feature, aggstat):
        ce  = ce_loss(pred, tgt)
        rl  = reg_loss(feature)

        loss = ce+ CFG.reg_coeff*rl

        loss_info = dict(**{"CE":ce.item(), "REG":rl.item()},)
        return loss, loss_info

    lutl.LOG2TXT(f"Using the Loss:::: CE + REG_{CFG.reg_method}",  CFG.gLogPath +'/misc.txt')
    return lossfunc


### ============================================================================


class ClsFedHandler(object):
    def __init__(self, trainloader, lossfunc, id=None,
                 validloader=None, device="cuda"):

        self.id          = id
        self.trainloader = trainloader
        self.validloader = validloader
        self.lossfunc    = lossfunc
        self.device      = device

        self.trainMetric = MultiClassMetrics(CFG.gLogPath+"/metrics/")
        self.validMetric = MultiClassMetrics(CFG.gLogPath+"/metrics/")
        self.loc_val_best = 0.0

        ##unused
        self.step_loader = iter(trainloader)
        self.local_optim = None
        self.local_scaler = None
        self.local_model = None
        self.aggstat     = None

    def train_one_epoch(self, epoch):

        model     = self.local_model
        optimizer = self.local_optim
        scheduler = self.local_scheduler
        scaler    = self.local_scaler

        if CFG.enable_fedprox:
            global_model_prox = copy.deepcopy(model)

        scheduler.last_epoch = epoch
        stat_accum = {}; locstat = {}
        model.train()
        ### ---------

        for step, (img, tgt) in tqdm(enumerate(self.trainloader,
                                            start=epoch*len(self.trainloader)
                                            ),
                                        disable=CFG.disable_tqdm):
            img = img.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)
            if img.shape[0] < CFG.batch_size: continue # fix last batch size being 1 issue

            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():
            pred = model.forward(img)
            loss, loss_info = self.lossfunc(pred, tgt, None, self.aggstat)

            if CFG.enable_fedprox:
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model_prox.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = loss+ (CFG.fedprox_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()
            # self.local_scaler.scale(loss).backward()
            # self.local_scaler.step(optimizer)
            # self.local_scaler.update()
            self.trainMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss, loss_info)

        #end epoch
        if scheduler: scheduler.step()

        model.eval()
        with torch.no_grad():
            for img, tgt in tqdm(self.validloader):
                img = img.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)
                if img.shape[0] < 2: continue # fix last batch size being 1 issue
                pred = model.forward(img)
                loss, loss_info = self.lossfunc(pred, tgt, None, self.aggstat)
                self.validMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss, loss_info)

        logs = dict(mode="Epoch-up", epoch=epoch, ID=self.id,
                    trainloss = self.trainMetric.get_loss(),
                    trainacc  = self.trainMetric.get_balanced_accuracy(),
                    trainF1   = self.trainMetric.get_f1score(),
                    validloss = self.validMetric.get_loss(),
                    validacc  = self.validMetric.get_balanced_accuracy(),
                    validF1   = self.validMetric.get_f1score(),
                    trainlossInfo = self.trainMetric.get_loss_info_aggregates(),
                    validlossInfo = self.validMetric.get_loss_info_aggregates(),
                    time=int(time.time()),)
        lutl.LOG2DICTXT(logs, CFG.gLogPath +'/train-local-stats.txt')
        lutl.LOG2CSV( [self.id,"#",epoch,"#"]+self.trainMetric.nnloss, CFG.gLogPath +'metrics/train-losses.csv')

        best_flag = False
        if self.loc_val_best < logs['validF1']:
            torch.save(model.state_dict(), CFG.gWeightPath +f'/best_local_model_{self.id}.pth')
            self.loc_val_best = logs['validF1']
            best_flag = True
            detail_stat = dict( ctime= time.ctime(),
                    ID=self.id, best = best_flag, epoch=epoch,
                    validbalacc = self.validMetric.get_balanced_accuracy(),
                    validf1scr  = self.validMetric.get_f1score(),
                    validreport = self.validMetric.get_class_report(),
                    # validconfus = validMetric.get_confusion_matrix().tolist(),
                )
            lutl.LOG2DICTXT(detail_stat, CFG.gLogPath+'/valid-local-bests.txt', console=False)

        self.trainMetric.reset()
        self.validMetric.reset()

        return model.state_dict(), {}


    def update_parameters(self, model, aggstat=None):
        self.local_model = model.to(self.device)
        self.local_optim = optim.AdamW(model.parameters(), lr=CFG.learning_rate,
                            weight_decay=CFG.weight_decay)
        self.local_scaler = torch.cuda.amp.GradScaler() # for mixed precision

        self.local_scheduler = None
        if CFG.enable_scheduler:
            self.local_scheduler = optim.lr_scheduler.MultiStepLR(self.local_optim,
                                    milestones=[int(CFG.epochs*0.5), int(CFG.epochs*0.75)],
                                    gamma=0.1)

        self.aggstat = aggstat

        self.local_optim.zero_grad()


### ----------------------------------------------------------------------------


def simple_main(model_key=None, folder_suffix=""):

    ### SETUP
    rutl.START_SEED()
    gpu_device = torch.device("cuda")
    torch.cuda.device(gpu_device)
    print("GPU device", gpu_device)

    # -- log path --
    if model_key: folder_suffix +=f"/{model_key}/"

    CFG.gLogPath = CFG.checkpoint_dir+folder_suffix
    CFG.gWeightPath = CFG.gLogPath+"/weights/"

    if os.path.exists(CFG.gLogPath) and (not CFG.resume_training):
        raise Exception("CheckPoint folder already exists and resume_training not enabled; Somethings Wrong!",
                        CFG.checkpoint_dir)
    if not os.path.exists(CFG.gWeightPath): os.makedirs(CFG.gWeightPath)

    save_current_configs(CFG)

    ### DATA ACCESS
    traindozers, validdozers = getDataLoaders(CFG, type="train")

    ### MODEL, OPTIM
    global_model = getModel(model_key, gpu_device)
    lossfn = getLossFunc()

    ## Automatically resume from checkpoint if it exists and enabled
    if os.path.exists(CFG.gWeightPath +'/checkpoint.pth') and CFG.resume_training:
        ckpt = torch.load(CFG.gWeightPath  +'/checkpoint.pth',
                            map_location='cpu')
        saved_global_round = ckpt['global_round']
        global_model.load_state_dict(ckpt['global_model'])
        lutl.LOG2TXT(f"Restarting Training from GSTEP:{saved_global_round} of {CFG.checkpoint_dir}",  CFG.gLogPath +'/misc.txt')
    else:
        saved_global_round = 0


    ### LOCAL SILOS setup
    aggstat = {}
    fed_locals = {}
    for id in traindozers.keys():
        fed_locals[id] = ClsFedHandler(id= id,
                                lossfunc = lossfn,
                                trainloader = traindozers[id],
                                validloader = validdozers[id],
                                )
        fed_locals[id].update_parameters(copy.deepcopy(global_model),
                                         aggstat=aggstat)


    if not CFG.enable_weight_reinit: lutl.LOG2TXT(("&"*7)+" Forgoing FedAveraging Routine ....", CFG.gLogPath +'/misc.txt')

    ### MODEL TRAINING
    print("Update Mode set::", CFG.update_mode)
    if CFG.update_mode == "step":
        raise Exception("Step Update Not implemented")
    elif CFG.update_mode == "epoch":
        total_itrs = CFG.global_rounds
        start_itrs = saved_global_round
    else: raise Exception("Unknown Update Mode set")


    start_time = time.time()
    best_acc = 0 ; best_loss = float('inf')

    for itr in range(start_itrs, total_itrs):

        ## ------ Training Routine ------
        local_weights, local_losses, local_astats = [], [], []
        global_model.train()

        for id in  traindozers.keys():
            ## run one epoch
            if CFG.update_mode == "epoch":
                if CFG.enable_weight_fedavg:
                    fed_locals[id].update_parameters(copy.deepcopy(global_model), aggstat = aggstat)
                else: fed_locals[id].aggstat = aggstat

                weight, astat = fed_locals[id].train_one_epoch(epoch = itr)

            else: raise Exception("Unknown Update Mode set")

            local_weights.append(copy.deepcopy(weight))
            local_astats.append(astat)

        global_weights = fedutl.global_average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # save checkpoint
        Gstep = (itr+1)/CFG.local_rounds if CFG.update_mode == "step" else itr

        if (Gstep+1) % CFG.ckpt_freq_Gstep == 0:
            Gstep = int(Gstep)
            state = dict(global_round=Gstep, global_model=global_model.state_dict())
            ## Local-Models
            #if not CFG.enable_weight_reinit: *->to save space
            for id in traindozers.keys():
                state[f"local_model_{id}"]= fed_locals[id].local_model.state_dict()
            torch.save(state, CFG.gWeightPath +f'/checkpoint.pth')


        ## ---- Global params Validation Routine ----
        globalValidMetric = MultiClassMetrics(CFG.gLogPath+"/metrics/")

        global_model.eval()
        with torch.no_grad():
            for img, tgt in tqdm(validdozers["all"]):
                img = img.to(gpu_device, non_blocking=True)
                tgt = tgt.to(gpu_device, non_blocking=True)
                pred = global_model.forward(img)
                loss, loss_info = lossfn(pred, tgt, None, aggstat)
                globalValidMetric.add_entry(torch.argmax(pred, dim=1),
                                            tgt, loss, loss_info)

        ## Log Metrics
        logs = dict(
                global_round=Gstep,
                run_time  = time.time()-start_time,
                validloss = globalValidMetric.get_loss(),
                validacc  = globalValidMetric.get_balanced_accuracy(),
                validF1   = globalValidMetric.get_f1score(),
                )
        lutl.LOG2DICTXT(logs, CFG.gLogPath+'/train-global-stats.txt')


        ## save best model
        best_flag = False
        if logs['validF1'] > best_acc:
            torch.save(global_model.state_dict(), CFG.gWeightPath +'/best_global_model.pth')
            best_acc  = logs['validF1']
            best_loss = logs['validloss']
            best_flag = True
            detail_stat = dict( ctime= time.ctime(),
                    epoch=Gstep, best = best_flag,
                    run_time=int(time.time() - start_time),
                    validbalacc = globalValidMetric.get_balanced_accuracy(),
                    validf1scr  = globalValidMetric.get_f1score(),
                    validreport = globalValidMetric.get_class_report(),
                    # validconfus = validMetric.get_confusion_matrix().tolist(),
                )
            lutl.LOG2DICTXT(detail_stat, CFG.gLogPath+'/valid-global-bests.txt', console=False)

    return CFG.gLogPath



def simple_test(saved_logpath, model_list=["global_model"]):

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

    for m in model_list:
        pth_list = {
            "best": torch.load(saved_logpath+f"/weights/best_{m}.pth"),
            "last": torch.load(saved_logpath+"/weights/checkpoint.pth")[f"{m}"],
        }

        ### MODEL TESTING
        for p_k, pth_wgt in pth_list.items():
            ret_msg = model.load_state_dict(pth_wgt, strict=False)
            lutl.LOG2TXT(f"Testing Weight Loaded...{CFG.featx_pretrain},{str(ret_msg)}; {p_k}--{saved_logpath} ",
                        CFG.gLogPath +'/misc.txt')

            test_center_num = CFG.test_partitions if CFG.test_partitions>1 else 0
            for c in ["all"]+ list(range(test_center_num)):
                testloader = getDataLoaders(CFG, center_index=c, type="test")
                testMetric = MultiClassMetrics(saved_logpath+ f"/metrics/{p_k}-test")
                model.eval()

                start_time = time.time()
                with torch.no_grad():
                    for img, tgt in tqdm(testloader, disable=CFG.disable_tqdm):
                        img = img.to(gpu_device, non_blocking=True)
                        tgt = tgt.to(gpu_device, non_blocking=True)
                        pred = model.forward(img)
                        testMetric.add_entry(torch.argmax(pred, dim=1), tgt)

                    ## Log detailed validation
                    log_title = f"test-{m}-{p_k}-{c}"
                    detail_logs = dict(
                            model_name  = m,
                            model_type  = p_k,
                            test_center = c,
                            timetaken   = int(time.time() - start_time),
                            ctime       = time.ctime(),
                            testf1scr   = testMetric.get_f1score(),
                            testbalacc  = testMetric.get_balanced_accuracy(),
                            testacc     = testMetric.get_accuracy(),
                            model_used  = os.path.basename(saved_logpath),
                            testreport  = testMetric.get_class_report(),
                            testconfus  = testMetric.get_confusion_matrix(
                                    save_png= True, title=log_title).tolist(),
                        )
                    lutl.LOG2DICTXT(detail_logs, saved_logpath+'/test-results.txt',
                                    console=True)

                    testMetric._write_predictions(title=log_title)



if __name__ == '__main__':


    model_list = ["global_model"]+[f"local_model_{i}"
                                   for i in range(CFG.data_centers_count)]


    if CFG.dataset == "CIFAR":
        quantity = [1000, 100, 10, 1, 0]
        for q in quantity:
            CFG.dirichlet_alpha = q
            qtitle = f"/{q}_aleph/"
            logpth = simple_main(folder_suffix=qtitle)
            simple_test(logpth, model_list)
    else:
        logpth = simple_main()
        simple_test(logpth, model_list)
