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
from utilities.metricUtils import MultiClassMetrics

import algorithms.federation_ops as fedops
import algorithms.federation_byz as fedbyz
from algorithms.classifier import ClassifierNet


print(f"Pytorch version: {torch.__version__}")
print(f"cuda version: {torch.version.cuda}")

##============================= Configure and Setup ============================

# Enable CODES for analysing model parameters and log their statitics
ANALYSE_MODELS = True
# To enable CODES for seperate validation routine based on local clients data
# user has to specify validation split in data csv, if not will yield empty results
VALIDATION = False

CFG = rutl.ObjDict(
dataset = "ISIC",
data_root_path  = "/home/joseph.benjamin/WERK/fed-cvpr/data/isic2019-jfed",
data_centers_count = 6,
test_partitions = 6,
center_inclusion_filter = [],
override_csv = None,
seed = 73,

## Fed specifics
dirichlet_alpha = None, # for Cifar100

global_rounds = 100, #Epochs
image_size    = 200,
batch_size    = 64,
workers       = 2,

learning_rate   = 5e-4,
weight_decay    = 1e-6,
enable_scheduler = True,

enable_weight_reinit = True, # FedAvg protocol, true in general cases
fed_approach = "fedavg",

enable_proxreg = False,
proxreg_mu     = 0.5,

reg_method   = "NONE",
reg_coeff    = 0.0,

featx_arch     = "resnet18",
featx_pretrain = "IMAGENET-1K" , # "IMAGENET-1K" or None``
featx_dropout  = 0.0,
featx_freeze   = False,
featx_bnorm    = False,

clsfy_layers   = [9], #First mlp inwill be set w.r.t FeatureExtractor
clsfy_dropout  = 0.0,

print_freq_lstep   = 0,
ckpt_freq_Gstep    = 1,
test_last_E_epochs = 5,  # detailed cross-client cross-data testing
test_trend_full    = False, # test with pooled test for all epochs

checkpoint_dir= "hypotheses/#dummy-run/trail-001",
resume_training=False
)

### -----
parser = argparse.ArgumentParser(description='Classification task')
parser.add_argument('--load-json', type=str, metavar='JSON',
    help='Load settings from file in json format which override values hard codes in py file.')

parser.add_argument('--seed', type=int, metavar='Int',
    help='Seed to be set for training')

parser.add_argument('--featx-pretrain', type=str, metavar='PATH',
    help='Set from where to load the prestrained weight from')

parser.add_argument('--checkpoint-dir', type=str, metavar='PATH',
    help='Load settings from file in json format. Command line options override values in file.')


args = parser.parse_args()

## update keys from json
if args.load_json:
    with open(args.load_json, 'rt') as f:
        CFG.__dict__.update(json.load(f))

## override json variables with CLI if any
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

    with open(os.path.realpath(fedbyz.__file__), "r") as pymain:
        file_contents += pymain.read()

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
    elif cfg.dataset == "ORGANMNIST":
        from datacode.orgmnist_jfed_data import getOrganMnistCLSLoaders as trainloader
        from datacode.orgmnist_jfed_data import getOrganMnistTESTLoader as testloader

    elif cfg.dataset == "HUMBLE":
        from datacode.humble_jfed_data import getHumbleCLSLoaders as trainloader
        from datacode.humble_jfed_data import getHumbleTESTLoader as testloader

    else:
        raise ValueError(f"Unsupported data type specfied {cfg.dataset}")

    if type =="train":
        traindozers = {}; validdozers = {}
        cen_list = range(cfg.data_centers_count) if not cfg.center_inclusion_filter else cfg.center_inclusion_filter
        for cen in list(cen_list):
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

def getFedProtocol():
    if CFG.fed_approach == "fedavg+state":
        fedProtocol = fedops.SimpleStateΞFedAvg

    elif CFG.fed_approach == "fedavg+byzantine":
        fedProtocol = fedbyz.NoGuardΞByzantine  #default
        fedProtocol = getattr(fedbyz, CFG.defense_cfg["defense_method"])

        if bool(CFG.center_inclusion_filter):
            raise Exception("Can't filter datacenters in Byzantine Method;; its a TODO") # remove dependency on CFG.data_centers

    else:
        raise Exception("Unknown Method given", CFG.fed_approach)
    return fedProtocol


def getLossFunc():
    # lossfn = WeightedFocalLoss(alpha=class_weights, gamma=0.2)
    ce_loss = nn.CrossEntropyLoss()

    if   CFG.reg_method == "TBD": reg_loss = lambda x: torch.tensor(0)
    else: reg_loss = lambda x: torch.tensor(0)

    def lossfunc(pred, tgt, feature, agghatch):
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
                 validloader=None, fedprtcl = None,
                 device="cuda"):

        self.id            = id
        self.trainloader   = trainloader
        self.validloader   = validloader
        self.lossfunc      = lossfunc
        self.device        = device
        self.fedprtcl      = fedprtcl

        self.num_class     = int(CFG.clsfy_layers[-1])
        self.trainMetric = MultiClassMetrics(CFG.gLogPath+"/metrics/")
        self.validMetric = MultiClassMetrics(CFG.gLogPath+"/metrics/")
        self.loc_val_best = 0.0

        self.step_loader  = iter(trainloader)
        self.local_optim  = None
        self.local_scaler = None
        self.local_model  = None # copy undergoing training
        self.gdsyp_model  = None # copy of model received fomr server
        self.winit_model  = None # copy of starting seed Model
        self.agghatch     = None

        ## TrainPhase attacks
        self.labelFlip=False
        if CFG.byztn_cfg["byztn_method"] == "LabelFlipζAttack":
            if self.id in CFG.byztn_cfg["byztn_clients"]:
                self.labelFlip = True


    def train_one_epoch(self, epoch):
        if self.labelFlip: print("LabelFLIPPING")

        model     = self.local_model
        optimizer = self.local_optim
        scheduler = self.local_scheduler
        scaler    = self.local_scaler
        return_result = {}

        if CFG.enable_proxreg:
            model_prox = copy.deepcopy(self.winit_model) ## --> change to

        if VALIDATION: startValidMetric = self.run_validation(model)
        ### --------------

        if scheduler: scheduler.last_epoch = epoch
        stat_accum = {}; locstat = {}

        model.train()
        for step, (img, tgt) in tqdm(enumerate(self.trainloader,
                                            start=epoch*len(self.trainloader)
                                            ),
                                        disable=CFG.disable_tqdm):
        # for step in tqdm(range(epoch*100, (epoch+1)*100, 1), disable=CFG.disable_tqdm):
        #     img, tgt = self.get_step_data()

            #------------------------------------------
            img = img.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)
            if img.shape[0] < 2: continue # fix last batch size being 1 issue

            if self.labelFlip: tgt = self.num_class - tgt -1; print("LF")

            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():
            pred, featp = model.forward(img)
            loss, loss_info = self.lossfunc(pred, tgt, featp, self.agghatch)

            if CFG.enable_proxreg:
                proximal_l2 = 0.0
                for w, w_t in zip(model.parameters(), model_prox.parameters()):
                    proximal_l2 += (w - w_t).norm(2).square()
                loss = loss + (CFG.proxreg_mu / 2) * proximal_l2

                loss_info["PROX_l2"] = ((CFG.proxreg_mu / 2) * proximal_l2).item()
                loss_info["PROX_cos"] = 0

            loss.backward()
            optimizer.step()
            # self.local_scaler.scale(loss).backward()
            # self.local_scaler.step(optimizer)
            # self.local_scaler.update()
            self.trainMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss, loss_info)
        #end epoch
        if scheduler: scheduler.step()

        ### --------------
        if VALIDATION: self.validMetric = self.run_validation(model)

        ### Logging
        logs = dict(mode="Epoch-up", epoch=epoch, ID=self.id,
                    trainloss = self.trainMetric.get_loss(),
                    trainacc  = self.trainMetric.get_balanced_accuracy(),
                    trainF1   = self.trainMetric.get_f1score(),
                    trainlossInfo = self.trainMetric.get_loss_info_aggregates(),
                    time=int(time.time())
                    )
        if VALIDATION:
            vallogs = dict(
                    validlossInfo = self.validMetric.get_loss_info_aggregates(),
                    validloss = self.validMetric.get_loss(),
                    validacc  = self.validMetric.get_balanced_accuracy(),
                    validF1   = self.validMetric.get_f1score(),
                    stValidloss = startValidMetric.get_loss(),
                    stValidacc  = startValidMetric.get_balanced_accuracy(),
                    stValidF1   = startValidMetric.get_f1score(),
                    )
            logs.update(vallogs)

        lutl.LOG2DICTXT(logs, CFG.gLogPath +'/train-local-stats.txt')
        lutl.LOG2CSV( [self.id,"#",epoch,"#"]+self.trainMetric.nnloss, CFG.gLogPath +'/metrics/train-losses.csv')

        ## best valiadation checkpoint
        if VALIDATION:
            best_flag = False
            if self.loc_val_best < logs['validF1']:
                # torch.save(model.state_dict(), CFG.gWeightPath +f'/best_local_model_{self.id}.pth') ##commenting since unused
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


        if ANALYSE_MODELS:
            model_start = self.winit_model
            curr_mvec = fedops.get_param_from_state(model.state_dict(), keys_to_ignore=["num_batches_tracked"])
            strt_mvec = fedops.get_param_from_state(model_start.state_dict(), keys_to_ignore=["num_batches_tracked"])
            model_diff_vec = curr_mvec - strt_mvec

            diff_dict ={"client": self.id, "epoch":epoch,
                        "delta_l2norm": torch.norm(model_diff_vec).item(),
                        "weight_cosim": torch_F.cosine_similarity(
                                curr_mvec.view(1,-1), strt_mvec.view(1,-1)).item(), }

            diff_dict["Layerwise"] =  fedutl.find_layerwise_weight_difference(model, model_start)
            lutl.LOG2DICTXT(diff_dict, CFG.gLogPath +'/trainAnsys-Wdiff_g-epochwise.txt', console=False)

            return_result["model_diff_vec"] = model_diff_vec
            return_result["model_vec"] = curr_mvec

        ## end >>>>> analyse_models

        self.trainMetric.reset()
        self.validMetric.reset()
        return_result["model"] = copy.deepcopy(model)
        return return_result


    def run_validation(self, model, prefix=""):
        tvalMetric = MultiClassMetrics(CFG.gLogPath+f"/metrics{prefix}/")
        model.eval()
        with torch.no_grad():
            for img, tgt in tqdm(self.validloader):
                img = img.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)
                if img.shape[0] < 2: continue # fix last batch size being 1 issue
                pred, featp = model.forward(img)
                loss, loss_info = self.lossfunc(pred, tgt, featp, self.agghatch)
                tvalMetric.add_entry(torch.argmax(pred, dim=1), tgt, loss, loss_info)

        return tvalMetric


    def update_parameters(self, model, agghatch=None):
        self.local_model = copy.deepcopy(model).to(self.device)
        self.gdsyp_model = copy.deepcopy(model).to(self.device).eval()
        if not self.winit_model: self.winit_model = copy.deepcopy(model).to(self.device).eval()

        self.local_optim = optim.AdamW(self.local_model.parameters(), lr=CFG.learning_rate,
                            weight_decay=CFG.weight_decay)
        self.local_scaler = torch.cuda.amp.GradScaler() # for mixed precision

        self.local_scheduler = None
        if CFG.enable_scheduler:
            self.local_scheduler = optim.lr_scheduler.MultiStepLR(self.local_optim,
                                    milestones=[int(CFG.global_rounds*0.5),
                                                int(CFG.global_rounds*0.75)],
                                    gamma=0.1)
        self.agghatch = agghatch

        self.local_optim.zero_grad()

    def get_step_data(self):
        try: step_data = next(self.step_loader)
        except StopIteration:
            self.step_loader = iter(self.trainloader)
            step_data = next(self.step_loader)

        return step_data

### ----------------------------------------------------------------------------


def simple_main(model_key=None, folder_suffix=""):

    ### SETUP
    print("SEED:", CFG.seed)
    rutl.START_SEED(CFG.seed)
    gpuid_generator = fedutl.gpu_devices_generator()

    # -- log path --
    if model_key: folder_suffix +=f"/{model_key}/"

    CFG.gLogPath = CFG.checkpoint_dir+folder_suffix
    CFG.gWeightPath = CFG.gLogPath+"/weights/"

    if os.path.exists(CFG.gLogPath) and (not CFG.resume_training):
        raise Exception("CheckPoint folder already exists and resume_training not enabled; Somethings Wrong!",
                        CFG.checkpoint_dir)
    if not os.path.exists(CFG.gWeightPath): os.makedirs(CFG.gWeightPath)

    save_current_configs(CFG)

    ### PROTOCOL
    fedProtocol = getFedProtocol()

    ### DATA ACCESS
    traindozers, validdozers = getDataLoaders(CFG, type="train")

    ### MODEL, OPTIM
    g_device = next(gpuid_generator)
    global_model = getModel(model_key, g_device)
    lossfn = getLossFunc()


    ## Automatically resume from checkpoint if it exists and enabled
    if os.path.exists(CFG.gWeightPath +'/checkpoint.pth') and CFG.resume_training:
        saved_global_round = 0
        print("regarding resuming training as Chadwick Boseman said `We Don't do that here`")

        # ckpt = torch.load(CFG.gWeightPath  +'/checkpoint.pth',
        #                     map_location='cpu')
        # saved_global_round = ckpt['global_round']
        # global_model.load_state_dict(ckpt['global_model'])
        # lutl.LOG2TXT(f"Restarting Training from GSTEP:{saved_global_round} of {CFG.checkpoint_dir}",  CFG.gLogPath +'/misc.txt')
    else:
        saved_global_round = 0

    #TODO: fix to run specific clients Federated training alone

    ### LOCAL SILOS setup
    agghatch        = None   # expanded/desynopsized information w.r.t local model
    global_agghatch = None   # expanded information w.r.t global model
    global_aggset   = None   # synopsized aggregate form global
    global_fedprtcl = fedProtocol(CFG, "G", global_model, g_device)
    fed_locals      = {}     # local Models hanger
    for id in traindozers.keys():
        l_device = next(gpuid_generator)
        fed_locals[id] = ClsFedHandler(id   = id,
                                lossfunc    = lossfn,
                                trainloader = traindozers[id],
                                validloader = validdozers[id],
                                fedprtcl    = fedProtocol(CFG, id, global_model, l_device),
                                device      = l_device
                                )
        fed_locals[id].update_parameters(global_model, agghatch=agghatch) #deepcopies inside

    if ANALYSE_MODELS:
        wvec_init =   fedops.get_param_from_state( global_model.state_dict(),
                                    keys_to_ignore=["num_batches_tracked"])

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
        local_xcerpt_for_fed = []
        local_train_returns = {}

        for id in  traindozers.keys():
            lmodel, agghatch = fed_locals[id].fedprtcl.desynopsize_local(global_aggset)
            # ## cached use for faster run; above method is right/ logic-bugfree !!!!!
            # lmodel, agghatch = global_model, global_agghatch

            ## run one epoch
            if CFG.update_mode == "epoch":
                if CFG.enable_weight_reinit:
                    fed_locals[id].update_parameters(lmodel, agghatch) #deepcopies inside
                else: fed_locals[id].agghatch = agghatch

                lret = fed_locals[id].train_one_epoch(epoch = itr)
            else: raise Exception("Unknown Update Mode set")

            local_train_returns[id] = lret #this was synopsize back then

        # modified synopsis for Omniscient attack
        for id in  traindozers.keys():
            syn_in = local_train_returns[id]
            syn_in["omniscience"] = local_train_returns
            local_xcerpt_for_fed.append(
                fed_locals[id].fedprtcl.synopsize_local(syn_in)  )

        global_aggset = global_fedprtcl.aggregate_globally(local_xcerpt_for_fed, device=g_device)

        ## caching to global_object for analysis
        global_model_tminus1 = copy.deepcopy(global_model)
        global_model, global_agghatch = global_fedprtcl.desynopsize_local(
                                                global_aggset, device=g_device,)

        if ANALYSE_MODELS:
            l2norm_dist  = []
            cosine_sim   = []
            diff_l2_norm = []
            diff_cos_sim = []
            winit_cos_sim = []
            for _, info1 in local_train_returns.items():
                cosim  = []
                l2nrm  = []
                difl2  = []
                difcos = []
                v1 = info1["model_vec"].to(g_device)
                dv1 = info1["model_diff_vec"].to(g_device)
                for _, info2 in local_train_returns.items():
                    dv2 = info2["model_diff_vec"].to(g_device)
                    v2 = info2["model_vec"].to(g_device)
                    l2nrm.append(torch.norm(v1-v2).item())
                    difl2.append(torch.norm(dv1-dv2).item())
                    cosim.append(torch_F.cosine_similarity(v1.view(1,-1), v2.view(1,-1)).item()  )
                    difcos.append(torch_F.cosine_similarity(dv1.view(1,-1), dv2.view(1,-1)).item()  )
                l2norm_dist.append(l2nrm)
                cosine_sim.append(cosim)
                diff_l2_norm.append(difl2)
                diff_cos_sim.append(difcos)
                winit_cos_sim.append(torch_F.cosine_similarity(v1.view(1,-1), wvec_init.view(1,-1)).item())
            dists_dict = {"epoch": itr, "l2norm":l2norm_dist, "cosim": cosine_sim,
                        "zero_cosim": winit_cos_sim,
                        "diff_l2norm":diff_l2_norm, "diff_cosim": diff_cos_sim}
            lutl.LOG2DICTXT(dists_dict, CFG.gLogPath +'/trainAnsys-weight-simMatrix.txt', console=False)

            cselect_dict = global_aggset.get("client_select")
            if cselect_dict:
                cselect_dict.update({"epoch":itr})
                lutl.LOG2DICTXT(cselect_dict, CFG.gLogPath +'/trainAnsys-client-selection.txt', console=False)
        ## end >>>>> analyse_models


        ## save checkpoint
        Gstep = (itr+1)//CFG.local_rounds if CFG.update_mode == "step" else itr

        if (Gstep+1) % CFG.ckpt_freq_Gstep == 0:
            state = dict(global_round=Gstep)
            state["global_model"] = global_model.state_dict() #updated after global round
            state["desyp_global_model"] = global_model_tminus1.state_dict() #global round begining
            ## Local-Models
            #if not CFG.enable_weight_reinit: *->to save space
            for id in traindozers.keys():
                state[f"local_model_{id}"] = fed_locals[id].local_model.state_dict() #updated after local round
                state[f"desyp_local_model_{id}"] = fed_locals[id].gdsyp_model.state_dict() #local round begining
            torch.save(state, CFG.gWeightPath +f'/checkpoint.pth')


        ## ---- Global params Validation Routine ----
        if VALIDATION:
            globalValidMetric = MultiClassMetrics(CFG.gLogPath+"/metrics/")

            global_model.eval()
            with torch.no_grad():
                for img, tgt in tqdm(validdozers["all"]):
                    img = img.to(g_device, non_blocking=True)
                    tgt = tgt.to(g_device, non_blocking=True)
                    pred, featp = global_model.forward(img)
                    loss, loss_info = lossfn(pred, tgt, featp, agghatch)
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
        ## end VALIDATION

        ## --------- Testing routines ---------------

        test_model_list = ["global_model"]

        ## Test every epoch for plotting
        if CFG.test_trend_full:
            simple_test(CFG.gLogPath, epochs_ran=itr,
                model_list=test_model_list,
                folder_suffix="", ## defaults to original folder
                ckpt_keys=["start"],
                test_partitions=0,
                console=False)

        test_model_list = ["global_model"]+[f"local_model_{id}"
                                for id in traindozers.keys()]

        ## Test Last N epochs for non fluctuating results
        if (CFG.test_last_E_epochs is not None ) and (CFG.update_mode == "epoch"):
            if (itr+1) > (CFG.global_rounds - CFG.test_last_E_epochs):
                print(test_model_list)
                simple_test(CFG.gLogPath, epochs_ran=itr,
                            model_list=test_model_list,
                            folder_suffix=f"test-epoch-{itr}",
                            ckpt_keys=["start", "last"],
                            test_partitions=CFG.test_partitions)


    return CFG.gLogPath



def simple_test(saved_logpath, model_list:list=["global_model"],
                epochs_ran=None, folder_suffix="",
                ckpt_keys:list = ["start"],
                test_partitions = CFG.test_partitions,
                console=True):

    gpu_device = torch.device("cuda")
    torch.cuda.device(gpu_device)

    dir_to_save = f"{saved_logpath}/{folder_suffix}/"
    os.makedirs(dir_to_save, exist_ok=True)

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
        pth_list = {}
        if "start" in ckpt_keys:
            pth_list["start"] = torch.load(saved_logpath+"/weights/checkpoint.pth")[f"desyp_{m}"]
        if "last" in ckpt_keys:
            pth_list["last"] = torch.load(saved_logpath+"/weights/checkpoint.pth")[f"{m}"]
        if "best" in ckpt_keys:
            pth_list["best"] = torch.load(saved_logpath+f"/weights/best_{m}.pth")

        ### MODEL TESTING
        for p_k in pth_list.keys():
            pth_wgt = pth_list[p_k]
            ret_msg = model.load_state_dict(pth_wgt, strict=False)
            lutl.LOG2TXT(f"Testing Weight Loaded...{CFG.featx_pretrain},{str(ret_msg)}; {p_k}--{saved_logpath} ",
                        dir_to_save +'/misc_test.txt')

            test_center_num = test_partitions if test_partitions>1 else 0
            for c in ["all"]+ list(range(test_center_num)):
                testloader = getDataLoaders(CFG, center_index=c, type="test")
                testMetric = MultiClassMetrics(dir_to_save+ f"/metrics/{p_k}-test")
                model.eval()

                start_time = time.time()
                with torch.no_grad():
                    for img, tgt in tqdm(testloader, disable=CFG.disable_tqdm):
                        img = img.to(gpu_device, non_blocking=True)
                        tgt = tgt.to(gpu_device, non_blocking=True)
                        pred,_ = model.forward(img)
                        testMetric.add_entry(torch.argmax(pred, dim=1), tgt)

                    ## Log detailed testing
                    log_title = f"test-{m}-{p_k}-{c}"
                    detail_logs = dict(
                            model_name  = m,
                            model_type  = p_k,
                            test_center = c,
                            epochs_ran  = epochs_ran,
                            timetaken   = int(time.time() - start_time),
                            seed        = CFG.seed,
                            ctime       = time.ctime(),
                            testf1scr   = testMetric.get_f1score(),
                            testbalacc  = testMetric.get_balanced_accuracy(),
                            testacc     = testMetric.get_accuracy(),
                            model_used  = os.path.basename(saved_logpath),
                            testreport  = testMetric.get_class_report(),
                            testconfus  = testMetric.get_confusion_matrix(
                                    save_png= True, title=log_title).tolist(),
                        )
                    lutl.LOG2DICTXT(detail_logs, dir_to_save+'/test-results.txt',
                                    console=console)

                    testMetric._write_predictions(title=log_title)



if __name__ == '__main__':


    def train_runner(model_key=None, folder_suffix=""):
        """Simple trainer"""
        test_model_list = ["global_model"]+[f"local_model_{i}"                  #==> Set as needed
                                for i in range(CFG.data_centers_count)]

        logpth = simple_main(model_key=model_key, folder_suffix=folder_suffix)
        # simple_test(logpth, model_list=test_model_list)

    ##-----

    def iided_vs_simple_wrap(xtitle=""):
        """For running different alephs/iidness in cifar100 or mnists"""

        if CFG.dataset == "CIFAR":
            quantity = [ 1000, 0, 100, 1, 10]                                   #==> Set as needed

            for q in quantity:
                CFG.dirichlet_alpha = q   #~~~~
                qtitle = xtitle + f"/{q}_aleph/"
                train_runner(folder_suffix=qtitle)

        elif CFG.dataset == "HUMBLE":
            quantity = ["full", "semi-mix", "semi-pure", "non"]

            for q in quantity:
                CFG.iid_ness = q     #~~~~
                qtitle = xtitle + f"/{q}_iid/"
                train_runner(folder_suffix=qtitle)

        else:
            train_runner()

    ##-----

    def sketch_compressions_wrap():

        ## HASHes set at 12
        compressions = {"1.5E": 8, #expand                                      #==> Set as needed
                        "2x": 24, "4x": 48, "8x": 96, "16x":198
                        }

        for cx, sx in compressions.items():
            CFG.sketch_compress_factor = sx  #~~~~
            xtitle = f"/{list(filter(None, CFG.checkpoint_dir.split('/')))[-1]}-{cx}/"
            print(xtitle)
            iided_vs_simple_wrap(xtitle)

    ###----------------------------------

    # train_runner()
    iided_vs_simple_wrap()
    # sketch_compressions_wrap()