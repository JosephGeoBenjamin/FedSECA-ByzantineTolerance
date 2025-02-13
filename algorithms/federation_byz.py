# Ξ - chi used as underscore in Classes

import copy
import h5py

import torch
import torch.nn.functional as torch_F
import numpy as np
import random
import scipy
import pyod
from sklearn.cluster import KMeans as skl_KMeans

import algorithms.federation_ops as fedops
import algorithms.byzantine_attacks as byz_attacks
import utilities.runUtils as rutl
from utilities.logUtils import LOG2CSV

"""
gmodel_init -> model recieved init communication i.e at very first broadcast of params for training
gmodel_tminus1 -> model recieved at (T-1)th aggregation round from Server
lmodel_tth -> gmodel_state_tminus1 updated for one set LocalRounds constituting Tth step
                before sending to Server

Tth: Ginit->L0->G0->L1->G1...->LT->GT->L
"""

##==============================================================================
## COMMONS

def get_attack_clsobj(method_string):
    attack_clsobj = getattr(byz_attacks, method_string)
    return attack_clsobj

def remove_diagonal(x): #contract along dim=1
    n, m = x.shape
    assert n == m
    x = x.flatten()[:-1].reshape(n - 1, n + 1)[:, 1:].flatten()
    x= x.reshape(n, n-1)
    return x

def insert_diagonal(x, D = 0.0): #expand along dim=1
    n, m = x.shape
    x = x.flatten().reshape(n - 1, n)
    x = np.hstack([ D*np.ones((n-1, 1)), x])
    x = np.hstack([x.flatten(), np.array([D])])
    x= x.reshape(n, n)
    return x

def torch_remove_diagonal(x): #contract along dim=1
    n, m = x.shape
    assert n == m
    x = x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    x = x.reshape(n, n-1)
    return x

def torch_insert_diagonal(x, D = 0.0): #expand along dim=1
    n, m = x.shape
    x = x.flatten().view(n - 1, n)
    x = torch.hstack([ D*torch.ones((n-1, 1)), x])
    x = torch.hstack([x.flatten(), torch.tensor([D])])
    x = x.reshape(n, n)
    return x

##==============================================================================
## Fed Protocol -- all are State Dict based aggregation

# FedAggregator = fedops.SimpleStateΞFedAvg

## **************************************

class NoGuardΞByzantine():
    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg

        self.gmodel_init  = copy.deepcopy(model).to(self.device) # common model initialization
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        self.aggregator_func = self.__plain_fedavg #override this to introduce methods
        print("DEFENSE: None")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()


    def _init_byzantiness(self):
        """NOTE: Attack class instantiation uses Dependency Injection / Callback pattern
        this is to enable dynamic attack optimization with knowledge of defence mechanism
        """
        byz_clients = [int(b) for b in self.byztn_cfg["byztn_clients"]]
        if self.id in byz_clients:
            self.byz_way = get_attack_clsobj(self.byztn_cfg["byztn_method"])(
                self.cfg, self.gmodel_init, defense_clsobj = self, device=self.device)

    def init_stacked_wvecs(self, model):
        pass

    #-------- Client methods ----------

    # @instancemethod #Locals calculation to send to Global L->G
    # used after One set local-rounds at each client
    def synopsize_local(self, zxs):
        """ zxs: {"model", }
        """
        lset = {}
        model = fedops.model_copier(zxs["model"])

        if self.byz_way:
            with torch.no_grad():
                omniinfo = zxs["omniscience"]
                omniinfo["self_K_id"] = self.id # ID for bookkeeping for cahooting
                out_state = self.byz_way.modify(model.state_dict(),
                                            self.gmodel_tminus1.state_dict(),
                                            omniscience=omniinfo)
        else:
            out_state = model.state_dict()

        lset["model_state"] = out_state

        return lset

    #-------- Shared methods ----------

    # @instancemethod # Process Global info for Local use G->L
    # used at end of Global Comm at each client
    def desynopsize_local(self, gset, device=None, model_struct=None):
        """ model_struct: torch nn.module object
            gset: global aggregations {"model", }
        """
        model_struct = self.gmodel_init if not model_struct else model_struct
        if not device: device = self.device

        ## since no compression or sketching used
        model = copy.deepcopy(model_struct)
        if not gset: return model, {}

        model.load_state_dict(copy.deepcopy(gset["model_state"]), strict=True)
        model = model.to(device)

        ghatch = {}
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device)

        return model, ghatch


    #-------- Server methods ----------

    def __plain_fedavg(self, lsets):
        # ## Regular
        # local_states = []
        # for ls in lsets:
        #     local_states.append(ls["model_state"])
        # agg_state = fedops.global_average_statedict(local_states, device="cpu")

        ## Vectorized
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        wvecs = [fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore = self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)

        agg_state = fedops.set_param_in_state(state_dict_struct, stacked_wvec.mean(dim=0),
                                              keys_to_ignore=self.vec_state_ignore)

        info_dict = {"client_weightage":[1/len(lsets)]*len(lsets)}
        return agg_state, info_dict


    # @instancemethod  #Global calculation to send to locals
    def aggregate_globally(self, lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        with torch.no_grad():
            agg_state, select_info = self.aggregator_func(lsets)

        gset = {"model_state": agg_state, "client_select": select_info}
        return gset



##==============================================================================

class KrumΞByzantine(NoGuardΞByzantine):
    """
    reference: Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent
    Paper: https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html

    """
    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        self.aggregator_func = self.__krum_aggregation
        self.krum_m  = int(self.defense_cfg["multikrum_m"]) # M ; number top clients to be used in averaging setup
        self.num_byz_b  = self.defense_cfg["num_assumed_byz"] # B; should hold 2b+1 <= n

        if not ( (2*self.num_byz_b+1) <= self.num_client_k):
            print(f"WARNING!!!..... `2b+1 <= n` doesn't hold, 2*{self.num_byz_b}+1 <= {self.num_client_k} ")
        print("Defense: Krum")


        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()


    #-------- Server methods ----------

    def __krum_aggregation(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        # fully_wvecs = [fedops.get_param_from_state(l["model_state"])
        #             for l in lsets]

        wvecs = [fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore = self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)
        K = len(lsets) # num_client_k

        num_neigbour = K - self.num_byz_b - 2 # assumed non malicious neighbours

        all_dist = []
        for v in wvecs:
            all_dist.append(torch.norm(stacked_wvec-v, dim=1).view(1,K))

        neighbor_dist_sum = []
        for dist in all_dist:
            vals, idxs = torch.topk(dist, k=num_neigbour+1, #since top will include self distance i.e zero
                       largest=False)
            neighbor_dist_sum.append(vals.sum())

        mvals, midxs = torch.topk(torch.hstack(neighbor_dist_sum),
                                  k=self.krum_m, largest=False)

        final_wvec = torch.zeros_like(wvecs[0])
        for mi in midxs.tolist():
            final_wvec +=wvecs[mi]
        final_wvec /= len(midxs)

        agg_state = fedops.set_param_in_state(state_dict_struct, final_wvec,
                                              keys_to_ignore=self.vec_state_ignore)

        info_dict = {"client_weightage": [ 1/len(midxs) if i in midxs else 0
                                        for i in range(len(lsets))]
                    }
        return agg_state, info_dict


##------------------------------------------------------------------------------

class CoordinateWiseCentralityΞByzantine(NoGuardΞByzantine):
    """
    reference: Dong Yin, et al. Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates
    Paper: https://proceedings.mlr.press/v80/yin18a.html

    Approaches: trimmedmean, median,
    """
    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        # number of byzzantines to ignore
        self.cwtm_beta = self.defense_cfg.get("cwtm_beta_count_oneside") #B; should hold K-2B > 0
        self.approach  = self.defense_cfg["approach"]
        self.aggregator_func = self.__coordinatewise_aggregation

        if self.num_client_k < (2 * self.cwtm_beta):
            raise Exception(f"Beta set is greater for given client count, 2*{self.cwtm_beta}>{self.num_client_k}")

        print(f"Defense: CoordinateWise {self.approach}")


        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()


    #-------- Server methods ----------

    def __coordinatewise_aggregation(self, lsets):
        """
        codes:
        https://github.com/moranant/attacking_distributed_learning/blob/master/defences.py
        https://github.com/epfml/byzantine-robust-optimizer/tree/main/codes/aggregator
        """
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])

        # fully_wvecs = [fedops.get_param_from_state(l["model_state"])
        #             for l in lsets]

        wvecs = [fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore = self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)
        K = len(lsets) # num_client_k

        median_wvec, _ = torch.median(stacked_wvec, dim=0)

        if self.approach == "median":
            final_wvec = median_wvec

        elif self.approach == "trimmedmean":

            # median_wvec = self.aggwvec_tminus ## to change median centring to wvec_tminus
            beta = self.cwtm_beta
            delta_wvec = stacked_wvec-median_wvec
            sorted_dwvec, _ = torch.sort(delta_wvec, dim=0)
            good_wvec = (sorted_dwvec[beta:][:])[:-beta][:]
            final_wvec = good_wvec.mean(dim=0) + median_wvec

        else: raise Exception(f"unknown method {self.approach}")

        agg_state = fedops.set_param_in_state(state_dict_struct,
                                final_wvec, keys_to_ignore=self.vec_state_ignore)

        info_dict = {"client_weightage": ["CW can't have this"]
                    }
        return agg_state, info_dict


##------------------------------------------------------------------------------

from pyod.models.copod import COPOD

class CopodDosΞByzantine(NoGuardΞByzantine):
    """
    reference: Suppressing Poisoning Attacks on Federated Learning for Medical Imaging
    paper: https://arxiv.org/abs/2207.10804
    """

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        self.aggregator_func = self.__dos_aggregation
        self.cpd_l2 = COPOD()
        self.cpd_cs = COPOD()

        print("Defense: Copod-DOS")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()



    #-------- Server methods ----------

    def __dos_aggregation(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        # fully_wvecs = [fedops.get_param_from_state(l["model_state"])
        #             for l in lsets]
        # sfully_wvec = torch.vstack(fully_wvecs)

        wvecs = [fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore = self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)

        l2_dists = []
        cs_dists = []
        for v in wvecs:
            l2_dists.append(torch.norm(stacked_wvec-v, dim=1))
            cs_dists.append(1-torch_F.cosine_similarity(stacked_wvec,
                                                      v.view(1, -1), dim=1))

        l2_dists = torch.nan_to_num(torch.vstack(l2_dists).cpu())
        cs_dists = torch.nan_to_num(torch.vstack(cs_dists).cpu())

        self.cpd_l2.fit(l2_dists)
        self.cpd_cs.fit(cs_dists)

        abnorm_score = (self.cpd_l2.decision_function(l2_dists) + \
                        self.cpd_cs.decision_function(cs_dists))
        abnorm_score = torch.tensor(abnorm_score).view(-1, 1)

        cweigh = torch_F.softmax(-1*abnorm_score,dim=0)
        cweighed_wvec = cweigh.to(self.device) * stacked_wvec  # s1*[v1] \ s2*[v2] \ s3*v3 ...

        final_wvec = torch.sum(cweighed_wvec, dim = 0)

        agg_state = fedops.set_param_in_state(state_dict_struct, final_wvec,
                                              keys_to_ignore=self.vec_state_ignore)

        info_dict = {"client_weightage":cweigh.flatten().tolist()}
        return agg_state, info_dict


##------------------------------------------------------------------------------

class GeoMedianRFAΞByzantine(NoGuardΞByzantine):
    """
    reference: Pillutla et al. Robust Aggregation for Federated Learning
    paper: https://arxiv.org/abs/1912.13445
    """

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        self.aggregator_func = self.__geomed_aggregation
        self.budget_R  = int(self.defense_cfg["budget_iter_r"]) # 3 in paper
        self.nu  = torch.tensor(self.defense_cfg["stability_nu"]) # 1e-6

        print("Defense: Geometric Median")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()


    #-------- Server methods ----------

    def __geomed_aggregation(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        # fully_wvecs = [fedops.get_param_from_state(l["model_state"]).to(self.device)
        #             for l in lsets]
        # sfully_wvec = torch.vstack(fully_wvecs)

        wvecs = [fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore = self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)

        # sample based weighting (α) is taken as constant 1/K, giving same weight for each clients
        alphas = torch.tensor([1 / len(lsets) for _ in lsets]).view(-1,1).to(self.device)
        betas_list = []


        ### Smoothed_Weiszfeld
        v = torch.zeros_like(wvecs[0])
        for r in range(self.budget_R):
            l2dist = torch.norm(v - stacked_wvec, dim=1).view(-1,1)
            betas = alphas / torch.maximum(l2dist, self.nu.to(self.device))

            v = betas * stacked_wvec # b1*[w1] \ b2*[w2] \ b3*[w3] ...
            v = v.sum(dim=0) / betas.sum(dim=0)

            betas_list.append(betas.flatten().tolist())
        ###

        final_wvec = v.clone()

        agg_state = fedops.set_param_in_state(state_dict_struct, final_wvec,
                                              keys_to_ignore=self.vec_state_ignore)

        info_dict = {"client_weightage":betas_list}
        return agg_state, info_dict


class HuberLossWeiszfeldΞByzantine(NoGuardΞByzantine):
    """ Minimization of Huberloss following Weiszfeild algorithm
    reference: Zhao, Puning, Fei Yu, and Zhiguo Wan. "A huber loss minimization approach to byzantine robust federated learning." AAAI-2024.
    paper: https://ojs.aaai.org/index.php/AAAI/article/view/30181
    """

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        self.aggregator_func = self.__huberloss_aggregation
        self.nu  = torch.tensor(self.defense_cfg["stability_nu"]) # 1e-6
        self.tolerance_tau  = float(self.defense_cfg["tolerance_iter_tau"]) #
        self.max_iters_R    = int(self.defense_cfg["max_iter_r"]) # for safety
        # TODO: make the huberloss_thresh dynamically computed based on dataset class
        self.threshold_T = float(self.defense_cfg["huberloss_thresh_t"]) # 2/sqrt(n_k) in paper, n_k is number of samples in i-th client

        print("Defense: Huber Loss with Weizsfield")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()


    #-------- Server methods ----------

    def __huberloss_aggregation(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        # fully_wvecs = [fedops.get_param_from_state(l["model_state"]).to(self.device)
        #             for l in lsets]
        # sfully_wvec = torch.vstack(fully_wvecs)

        wvecs = [fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore = self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)

        betas_list = []

        ### Weiszfeld like huberloss minimization
        c = torch.zeros_like(wvecs[0])
        c_prv = c.clone()
        for r in range(self.max_iters_R):
            l2dist = torch.norm(c - stacked_wvec, dim=1).view(-1,1)
            betas_raw = self.threshold_T / torch.maximum(l2dist, self.nu.to(self.device))
            betas = torch.minimum(torch.tensor(1.0).to(self.device), betas_raw)

            c = betas * stacked_wvec # b1*[w1] \ b2*[w2] \ b3*[w3] ...
            c = c.sum(dim=0) / betas.sum(dim=0)

            betas_list.append(betas.flatten().tolist())

            if torch.norm(c - c_prv).item() < self.tolerance_tau: break
            else: c_prv = c.clone()

        ###
        final_wvec = c.clone()

        agg_state = fedops.set_param_in_state(state_dict_struct, final_wvec,
                                              keys_to_ignore=self.vec_state_ignore)

        info_dict = {"client_weightage":betas_list}
        return agg_state, info_dict




##==============================================================================

class ClippingΞByzantine(NoGuardΞByzantine):
    """
    reference: Karimireddy et al. "Learning from history for byzantine robust optimization." ICML2021
    """

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = K = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]


        self.clip_iters = int(self.defense_cfg.get("clip_iters")) # if zero no clipping will happen
        if self.clip_iters==0: print("Clipping disabled since clip iters is 0")
        self.radius_estimate = self.defense_cfg.get("clip_radius")

        self.aggregator_func = self.__clipping_aggregate

        ##
        self.vec_state_ignore = ["num_batches_tracked"] # critical for l2norms since this skews it
        if self.id == "G": #large tensors, so why waste mem
            self.init_stacked_wvecs(model)

        print("Defense: Clipping")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------

    def init_stacked_wvecs(self, model):
        wvec = fedops.get_param_from_state(model.state_dict(),
                        keys_to_ignore=self.vec_state_ignore).to(self.device)
        self.aggwvec_tminus1 = wvec
        self.mom_deltawvec = torch.zeros_like(wvec)


    def get_tau(self):
        dtype_ = self.aggwvec_tminus1.dtype

        tau = self.radius_estimate
        return torch.tensor(tau, dtype=dtype_).view(1).to(self.device)


    #-------- Server methods ----------
    def safe_divide(self, nu, de, fill=1.0):
        res = torch.full_like(de, fill_value=fill)
        mask = (de != 0.0)

        if (nu.shape == mask.shape): nu_ = nu[mask]
        elif (sum(nu.shape) == 1):   nu_ = nu
        else: raise Exception(f"Incompatible shapes {de.shape}, {nu.shape}")

        res[mask] = torch.div(nu_, de[mask])
        return res

    def clipping_operation(self, x, v, tau, c_iter=1):
        """
        x : vectors   :shp:[K, len_parameters]
        v : reference estimate vector   :shp:[1, len_parameters]
        tau: clipping radius
        """
        rad_info = []
        for i in range(c_iter):
            stacked_gdelta = torch.norm(x-v, dim=1).view(-1, 1) #

            tau_by_ccden = self.safe_divide(tau, stacked_gdelta, fill=0.0) ## just setting clipping radius for zero norm
            rad_comp = torch.minimum(torch.tensor(1), tau_by_ccden).view(-1,1)

            clipped_delta = rad_comp * (x - v) # s1*[v1] \ s2*[v2] \ s3*v3 ...
            clipped_aggdelta = torch.mean(clipped_delta, dim = 0)
            v = v + clipped_aggdelta

            rad_info.append(rad_comp.flatten().tolist())

        return v, rad_info


    def __clipping_aggregate(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        K = len(lsets)

        #for l2norms
        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)
        stacked_deltawvec = self.aggwvec_tminus1 - stacked_wvec

        tau = self.get_tau()

        ## Clipping
        self.mom_deltawvec, rad_info = self.clipping_operation(stacked_deltawvec,
                                                     self.mom_deltawvec,
                                                     tau, self.clip_iters)

        new_wvec = self.aggwvec_tminus1 - self.mom_deltawvec
        agg_state = fedops.set_param_in_state(state_dict_struct, new_wvec,
                                               keys_to_ignore=self.vec_state_ignore)

        self.aggwvec_tminus1 = new_wvec.clone()

        info_dict = {"client_clip_weightage":rad_info}

        return agg_state, info_dict


class RandomBucketingΞByzantine(ClippingΞByzantine):
    """
    reference: Karimireddy et al. "Byzantine-robust learning on heterogeneous datasets via bucketing." ICLR2022.
    """
    def __init__(self, cfg, id, model, device="cpu"):
        super().__init__(cfg, id, model, device)

        self.bucket_size = int(self.defense_cfg.get("bucket_size_s")) # S

        self.aggregator_func = self.__random_bucket_aggregate

        print(" WITH RandomBucketing ")

    def _random_subsets(self, indices, subset_size):
        indices = random.sample(indices, len(indices))
        out = []
        for i in range(0, len(indices), subset_size):
            out.append( indices[i:i+subset_size] )
        return out


    def __random_bucket_aggregate(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        K = len(lsets)

        #for l2norms
        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs);    del wvecs
        stacked_deltawvec = self.aggwvec_tminus1 - stacked_wvec

        tau = self.get_tau()

        ## Bucketing
        grps = self._random_subsets(list(range(K)), self.bucket_size)
        bucketed_list = []
        for gp in grps:
            bucket_mean = stacked_deltawvec[gp,:].mean(dim=0).view(1,-1)
            bucketed_list.append(bucket_mean)
        bucketed_deltawvec = torch.vstack(bucketed_list);    del bucketed_list

        ## Clipping
        self.mom_deltawvec, rad_info = self.clipping_operation(bucketed_deltawvec,
                                                     self.mom_deltawvec,
                                                     tau, self.clip_iters)

        new_wvec = self.aggwvec_tminus1 - self.mom_deltawvec
        agg_state = fedops.set_param_in_state(state_dict_struct, new_wvec,
                                               keys_to_ignore=self.vec_state_ignore)

        self.aggwvec_tminus1 = new_wvec.clone()

        info_dict = {"client_clip_weightage":rad_info}

        return agg_state, info_dict

class SequentialBucketingΞByzantine(ClippingΞByzantine):
    """
    reference: Ozfatura et al. Byzantines can also Learn from History: Fall of Centered Clipping in Federated Learning
    """
    def __init__(self, cfg, id, model, device="cpu"):
        super().__init__(cfg, id, model, device)

        self.bucket_size = int(self.defense_cfg.get("bucket_size_s")) # S

        self.aggregator_func = self.__sequential_bucket_aggregate

        print(" WITH SequentialBucketing ")

    def _cosine_sorted_subsets(self, cosine_vals:torch.tensor, subset_size):
        _, indices = torch.sort(cosine_vals)
        indices = indices.tolist()

        l = np.ceil(len(indices)/subset_size).astype(int) #num_bucket
        # get S clusters
        clusters = [ random.sample(indices[i*l: (i+1)*l], len(indices[i*l: (i+1)*l]))
                                   for i in range(subset_size)]

        out = []
        for i in range(0, l):
            buck_ = []
            for j in range(subset_size):
                buck_.extend(clusters[j][i:i+1])
            out.append(buck_)
        return out


    def __sequential_bucket_aggregate(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        K = len(lsets)

        #for l2norms
        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs);    del wvecs
        stacked_deltawvec = self.aggwvec_tminus1 - stacked_wvec

        tau = self.get_tau()

        ## Bucketing
        cos_scores = torch_F.cosine_similarity( self.mom_deltawvec,
                                    stacked_deltawvec, dim=1).flatten()
        grps = self._cosine_sorted_subsets(cos_scores, self.bucket_size)
        info_list = []
        for gp in grps:
            bucket_wvec = stacked_deltawvec[gp,:].view(len(gp), -1)
            self.mom_deltawvec, rad_info = self.clipping_operation(bucket_wvec,
                                                        self.mom_deltawvec,
                                                        tau, self.clip_iters)
            info_list.append(rad_info)

        new_wvec = self.aggwvec_tminus1 - self.mom_deltawvec
        agg_state = fedops.set_param_in_state(state_dict_struct, new_wvec,
                                               keys_to_ignore=self.vec_state_ignore)

        self.aggwvec_tminus1 = new_wvec.clone()

        info_dict = {"client_clip_weightage":rad_info}

        return agg_state, info_dict


class FedNGAΞByzantine(NoGuardΞByzantine):
    """ Normalized Gradient Aggregation
    Reference: Zuo, Shiyuan, et al. "Byzantine-resilient Federated Learning Employing Normalized Gradients on Non-IID Datasets."
    paper: https://arxiv.org/abs/2408.09539v1
    """

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = K = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]


        self.aggregator_func = self.__normalized_grad_aggregate

        ##
        self.vec_state_ignore = ["num_batches_tracked"] # critical for l2norms since this skews it
        if self.id == "G": #large tensors, so why waste mem
            self.init_stacked_wvecs(model)

        print("Defense: Fed-NormGradAgg")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------

    def init_stacked_wvecs(self, model):
        wvec = fedops.get_param_from_state(model.state_dict(),
                        keys_to_ignore=self.vec_state_ignore).to(self.device)
        self.aggwvec_tminus1 = wvec
        self.mom_deltawvec = torch.zeros_like(wvec)


    #-------- Server methods ----------
    def safe_divide(self, nu, de, fill=1.0):
        res = torch.full_like(de, fill_value=fill)
        mask = (de != 0.0)

        if (nu.shape == mask.shape): nu_ = nu[mask]
        elif (sum(nu.shape) == 1):   nu_ = nu
        else: raise Exception(f"Incompatible shapes {de.shape}, {nu.shape}")

        res[mask] = torch.div(nu_, de[mask])
        return res



    def __normalized_grad_aggregate(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        K = len(lsets)

        #for l2norms
        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)
        stacked_deltawvec = self.aggwvec_tminus1 - stacked_wvec


        ## Normalise
        stacked_norms = torch.norm(stacked_deltawvec, dim=1).view(-1, 1)
        new_deltawvec = (stacked_deltawvec / stacked_norms).mean(dim=0)

        new_wvec = self.aggwvec_tminus1 - new_deltawvec
        agg_state = fedops.set_param_in_state(state_dict_struct, new_wvec,
                                               keys_to_ignore=self.vec_state_ignore)

        self.aggwvec_tminus1 = new_wvec.clone()

        info_dict = {"client_norm_values": stacked_norms.cpu().tolist()}

        return agg_state, info_dict

##==============================================================================

class FLDetectorΞByzantine(NoGuardΞByzantine):
    """
    reference: Zhang, Zaixi, et al. "Fldetector: Defending federated learning against model poisoning attacks via detecting malicious clients." SIGKDD 2022.
    """

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = K = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        self.window_n = int(self.defense_cfg.get("history_window")) # N in paper
        self.aggr_method = self.defense_cfg.get("aggregator_method")
        assert self.aggr_method == "fedavg", f"Expected 'fedavg' but got '{self.aggr_method}'"

        self.aggregator_func = self.__byzants_detector

        ##
        self.vec_state_ignore = ["num_batches_tracked"] # critical for l2norms since this skews it
        if self.id == "G": #large tensors, so why waste mem
            self.init_stacked_wvecs(model)

        print(f"Defense: FL Detector w/ {self.aggr_method}")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------

    def init_stacked_wvecs(self, model):
        wvec = fedops.get_param_from_state(model.state_dict(),
                        keys_to_ignore=self.vec_state_ignore).to(self.device)
        self.aggwvec_tminus1 = wvec
        self.aggdeltavec_tminus1 = torch.zeros_like(wvec)

        self.model_diffs = [] # ΔW of global
        self.update_diffs = [] # ΔG of global
        self.edist_priors = []

    #-------- Server methods ----------
    def safe_divide(self, nu, de, fill=1.0):
        res = torch.full_like(nu, fill_value=fill)
        mask = (de != 0.0)

        if (nu.shape == mask.shape): nu_ = nu[mask]
        elif (sum(nu.shape) == 1):   nu_ = nu
        else: raise Exception(f"Incompatible shapes {de.shape}, {nu.shape}")

        res[mask] = torch.div(nu_, de[mask])
        return res


    def _hessian_vector_product(self, dW_t_list, dG_t_list, stacked_dV):
        """ Using L-BFGS
        list of 1xD vectors, len is equal to window size (WN)
        """
        stacked_dV = stacked_dV.T # D x Klients
        dW_t = torch.vstack(dW_t_list).T #  D x WN
        dG_t = torch.vstack(dG_t_list).T #  D x WN

        dW_t_time_dG_t = torch.matmul(dW_t.T, dG_t)
        dW_t_time_dW_t = torch.matmul(dW_t.T, dW_t)

        # Extract upper triangular part of dW_k_time_dG_k -> [WN, WN]
        R_t = torch.triu(dW_t_time_dG_t)
        L_t = dW_t_time_dG_t - R_t.to(self.device)  # [WN, WN]

        sigma_k = torch.dot(dG_t_list[-1], dW_t_list[-1]) / torch.dot(dW_t_list[-1], dW_t_list[-1])  # []
        D_t_diag = torch.diag(dW_t_time_dG_t)  # [WN]

        upper_mat = torch.cat([sigma_k * dW_t_time_dW_t, L_t], dim=1)  # [WN, 2WN]
        lower_mat = torch.cat([L_t.t(), -torch.diag(D_t_diag)], dim=1)  # [WN, 2WN]

        mat = torch.cat([upper_mat, lower_mat], dim=0)  # [2WN, 2WN]
        mat_inv = torch.linalg.inv(mat)  # [2WN, 2WN]

        approx_prod = sigma_k * stacked_dV # [D x K]

        p_mat = torch.cat([torch.matmul(dW_t.T, sigma_k * stacked_dV), torch.matmul(dG_t.T, stacked_dV)], dim=0)  # [2WN x K]
        approx_prod -= torch.matmul(torch.matmul(torch.cat([sigma_k * dW_t, dG_t], dim=1), mat_inv), p_mat)  # [D x K]

        return approx_prod.T


    def _compute_suspicion_score(self, g_estim, g_actual):
        edist_t = (g_estim - g_actual).norm(dim=1).view(-1,1) # K x 1
        self.edist_priors.append(edist_t)

        sscores_t = torch.stack(self.edist_priors, dim=1).mean(dim=1) #K x 1

        if len(self.edist_priors) >= self.window_n: #NOTE: just lazy coding
            del self.edist_priors[0]

        return sscores_t


    def _detect_malicious_clients(self, score):

        atk_clients = []

        ## GAP analysis
        nrefs = 10
        ks = range(1, np.min([self.num_client_k, 25]))
        gaps = np.zeros(len(ks))
        gapDiff = np.zeros(len(ks) - 1)
        sdk = np.zeros(len(ks))

        score[np.isnan(score)] = 0
        min = np.min(score)
        max = np.max(score)

        # guard when scores are same for all, return empty
        if np.isclose(max-min, 0.0): return atk_clients

        try: # TODO: sometimes scores hit nan even after handling above, debug that
            score = (score - min)/(max-min)
            for i, k in enumerate(ks):
                estimator = skl_KMeans(n_clusters=k)
                estimator.fit(score.reshape(-1, 1))
                label_pred = estimator.labels_
                center = estimator.cluster_centers_
                Wk = np.sum([np.square(score[m]-center[label_pred[m]]) for m in range(len(score))])
                WkRef = np.zeros(nrefs)
                for j in range(nrefs):
                    rand = np.random.uniform(0, 1, len(score))
                    estimator = skl_KMeans(n_clusters=k)
                    estimator.fit(rand.reshape(-1, 1))
                    label_pred = estimator.labels_
                    center = estimator.cluster_centers_
                    WkRef[j] = np.sum([np.square(rand[m]-center[label_pred[m]]) for m in range(len(rand))])
                gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
                sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))
                if i > 0:
                    gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
        except:
            print("Unstable KMeans Clutering, returning empty")
            return atk_clients

        select_k = 1
        for i in range(len(gapDiff)):
            if gapDiff[i] >= 0:
                select_k = i+1
                break

        ## Find if Malicious Clients exists
        if select_k == 1:
            print('FLDetect: No attack detected!')
        else:
            estimator = skl_KMeans(n_clusters=2)
            estimator.fit(score.reshape(-1, 1))
            label_pred = estimator.labels_
            # 0 is taken as label of malicious clients
            if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
                label_pred = 1 - label_pred # change 0 assignment based on score sum
            atk_clients = np.where(label_pred==0)
            print(f'FLDetect: Attackers are {atk_clients}')

        return atk_clients


    def __byzants_detector(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        K = len(lsets)

        #for l2norms
        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)
        stacked_deltawvec = self.aggwvec_tminus1 - stacked_wvec # stacked_dV

        ###------- check if epoch is more than 0
        nonmalicious_size = self.num_client_k
        mal_clients = []
        if len(self.model_diffs) > 0:
            sdeltawvec_cap = self._hessian_vector_product(self.model_diffs,
                                                        self.update_diffs,
                                                        stacked_deltawvec)

            susp_scores = self._compute_suspicion_score(stacked_deltawvec, sdeltawvec_cap)

            mal_clients = self._detect_malicious_clients(susp_scores.cpu().numpy())
            nonmalicious_size = self.num_client_k - len(mal_clients)

            mal_idx = torch.tensor(mal_clients).to(self.device,dtype=torch.long).view(-1)
            stacked_deltawvec[mal_idx, :] = 0


        ## FedAVG on selected clients
        # raggr_dwvec = stacked_deltawvec.sum(dim=0) / nonmalicious_size
        ## Fed Median on sleted clients
        raggr_dwvec, _ = torch.median(stacked_deltawvec, dim=0)

        new_wvec = self.aggwvec_tminus1 - raggr_dwvec
        agg_state = fedops.set_param_in_state(state_dict_struct, new_wvec,
                                               keys_to_ignore=self.vec_state_ignore)

        self.model_diffs.append(raggr_dwvec.clone())
        self.update_diffs.append(self.aggdeltavec_tminus1 - raggr_dwvec)

        self.aggdeltavec_tminus1 = raggr_dwvec.clone()
        self.aggwvec_tminus1 = new_wvec.clone()

        if len(self.model_diffs) > self.window_n: del self.model_diffs[0]
        if len(self.update_diffs) > self.window_n: del self.update_diffs[0]

        info_dict = {"malicious_client": mal_clients}

        return agg_state, info_dict


##==============================================================================

class TiesMergeΞByzantine(NoGuardΞByzantine):
    """
    reference: Yadav et al. Ties-merging: Resolving interference when merging models - Neurips2024 .
    """

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = K = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        self.tm_beta = self.defense_cfg.get("tm_beta")

        self.aggregator_func = self.__ties_merging

        ##
        self.vec_state_ignore = ["num_batches_tracked"] # critical for l2norms since this skews it
        if self.id == "G": #large tensors, so why waste mem
            self.init_stacked_wvecs(model)

        print("Defense: Ties Merging")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------

    def init_stacked_wvecs(self, model):
        wvec = fedops.get_param_from_state(model.state_dict(),
                        keys_to_ignore=self.vec_state_ignore).to(self.device)
        self.aggwvec_tminus1 = wvec


    #-------- Server methods ----------
    def safe_divide(self, nu, de, fill=1.0):
        res = torch.full_like(nu, fill_value=fill)
        mask = (de != 0.0)

        if (nu.shape == mask.shape): nu_ = nu[mask]
        elif (sum(nu.shape) == 1):   nu_ = nu
        else: raise Exception(f"Incompatible shapes {de.shape}, {nu.shape}")

        res[mask] = torch.div(nu_, de[mask])
        return res

    def tensor_quantile(self, tnsr, q, dim=1):
        # torch quantile only works for 16M<elements
        numpy_tensor = tnsr.cpu().numpy()
        result = np.quantile(numpy_tensor, q, axis=dim)
        tnsr_result = torch.tensor(result).to(tnsr.device)
        return tnsr_result



    def __ties_merging(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        K = len(lsets)

        #for l2norms
        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)
        stacked_deltawvec = self.aggwvec_tminus1 - stacked_wvec

        # for mag and sgn vectors
        magn_dwvec = torch.abs(stacked_deltawvec).view(K,-1)

        qs = self.tensor_quantile(magn_dwvec, self.tm_beta, dim=1)
        qs = qs.view(-1, 1)
        stacked_deltawvec[magn_dwvec<qs] = 0.0

        sign_dwvec = torch.sign(stacked_deltawvec.sum(dim=0)).view(1,-1)


        disjoint_select = (0<(stacked_deltawvec * sign_dwvec)).bool() #select similar signed values

        disjoint_dwvec = stacked_deltawvec * disjoint_select
        disjoint_divisor = disjoint_select.sum(dim=0)

        mean_dwvec = self.safe_divide(disjoint_dwvec.sum(dim=0), disjoint_divisor, fill=0.0)


        new_wvec = self.aggwvec_tminus1 - mean_dwvec
        agg_state = fedops.set_param_in_state(state_dict_struct, new_wvec,
                                               keys_to_ignore=self.vec_state_ignore)

        self.aggwvec_tminus1 = new_wvec.clone()

        info_dict = {"client_clip_weightage": "Nope Can't do for coordwise operation"}

        return agg_state, info_dict


##==============================================================================

##******************************************************************************

#FedSECA
class FedSECAΞByzantine(NoGuardΞByzantine):
    """ This is FedSECA implementation which includes CRISE and ROCA steps
    Previously aliased FedRiseV2, anywhere it says this it points to FedSECA
    This is an improvement on thesis work titled FedRISE.
    """

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = K = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        self.mom_beta = self.defense_cfg.get("moment_beta")
        self.tm_gamma = self.defense_cfg.get("tm_gamma")

        self.aggregator_func = self.__fedrise_merging

        ##
        self.vec_state_ignore = ["num_batches_tracked"] # critical for l2norms since this skews it
        if self.id == "G": #large tensors, so why waste mem
            self.init_stacked_wvecs(model)

        print(f"Defense: FedSECA beta-{self.mom_beta} gamma-{self.tm_gamma}")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------
    def init_stacked_wvecs(self, model):
        wvec = fedops.get_param_from_state(model.state_dict(),
                        keys_to_ignore=self.vec_state_ignore).to(self.device)
        self.aggwvec_tminus1 = wvec
        self.mom_deltawvec = torch.zeros_like(wvec).to(self.device)
        # self.error_deltawvec = torch.zeros_like(wvec)
        # self.prior_repute = 0

    # ---------------
    # Server methods -----------------------------------------------------------
    # ---------------

    def safe_divide(self, nu, de, fill=1.0): ## shape Fixed version
        res_like = de if (sum(nu.shape) < sum(de.shape)) else nu
        res = torch.full_like(res_like, fill_value=fill)

        try:
            nu, de = torch.broadcast_tensors(nu, de)
        except Exception as err:
            raise Exception(f"Incompatible shapes {de.shape}, {nu.shape} -- \n{err}")

        mask = (de != 0.0)
        res[mask] = torch.div(nu[mask], de[mask])
        return res

    def tensor_quantile(self, tnsr, q, dim=1):
        # torch quantile only works for 16M<elements
        numpy_tensor = tnsr.cpu().numpy()
        result = np.quantile(numpy_tensor, q, axis=dim)
        tnsr_result = torch.tensor(result).to(tnsr.device)
        return tnsr_result

    # ------------ Gradient Clipping -------------------------------------------

    def _locwise_grad_clamper(self, xs):
        """
        Norm Clipping & Median Clamping
        x : vectors   :shp:[K, len_parameters]
        """

        norms = torch.norm(xs, dim=1).view(-1, 1)
        med_norm, _ = torch.median(norms, dim=0)
        norm_clip = med_norm/norms
        norm_clip[norm_clip>1.0] = 1.0
        xs_clipped = xs * norm_clip.view(-1, 1)

        med_mag,_ = torch.median(torch.abs(xs_clipped), dim=0)
        vs = torch.clamp(xs, max=med_mag, min=-med_mag)

        return vs


    ## ----------------------- Reputation --------------------------------------

    def _torch_Cordancy(self, a, b):
        """
        Kendall Formulation
        tau_a = (P - Q) / (N(N-1)/2)
        tau_b = (P - Q) / sqrt((P + Q + T) * (P + Q + U))
        """

        a_sgn = torch.sign(a)
        b_sgn = torch.sign(b)

        sgn_pair = a_sgn * b_sgn

        taua = sgn_pair.sum(dim=1) # unnormalised

        # n_conc = (sgn_pair>0).sum(dim=1)
        # n_disc = (sgn_pair<0).sum(dim=1)
        # n = torch.prod(torch.tensor(b_sgn[0].shape)) #number of params
        # taua = (n_conc - n_disc) / (n_conc+n_disc)


        ## taua = (n_conc - n_disc) / torch.prod(torch.tensor(a_sgn.shape))
        ## taua = (n_conc - n_disc) / (n*(n-1)/2)

        return taua


    def _grad_rating_score(self, sign_x):

        score_list = []
        for i in range(sign_x.shape[0]):
            score = torch_F.cosine_similarity(sign_x , sign_x[i].view(1,-1))
            # score = self._torch_Cordancy(sign_x , sign_x[i].view(1,-1))
            s = torch.sign(score).mean()
            score_list.append(s)
        current_repute = torch.vstack(score_list)

        repute = torch.clamp(current_repute, min=0)
        # repute = torch_F.softmax(repute, dim=0)
        return repute


    ## ----------------------- Merging --------------------------------------

    def sign_voted_mean(self, dw):
        x_raw = dw

        ## clamp the max grads
        x = self._locwise_grad_clamper(x_raw)

        ## mag and sgn vectors -> for ties
        magn_xraw = torch.abs(x_raw)

        ##NOTE: Time complexity of this Top-K sparsification could be improved
        ##      by followed by technique used in "Deep Gradient Compression" paper
        ql = self.tensor_quantile(magn_xraw, self.tm_gamma, dim=1)
        ql = ql.view(-1, 1)
        x[magn_xraw<ql] = 0.0

        ## vote with repute
        sign_xraw = torch.sign(x_raw)
        repute = self._grad_rating_score(sign_xraw)

        sign_x = torch.sign(x)
        voted_sign = (sign_x*repute.view(-1,1)).sum(dim=0).view(1,-1)

        disjoint_select = (0<(x * voted_sign)).int() #select similar signed values
        disjoint_x = x * disjoint_select
        disjoint_divisor = disjoint_select.sum(dim=0)

        ## mean final output
        mean_dwvec = self.safe_divide(disjoint_x.sum(dim=0), disjoint_divisor, fill=0.0)

        print("\n\n\n", repute,"\n", disjoint_select.sum(dim=1).tolist())
        return mean_dwvec.view(1,-1), repute.flatten().tolist()



    def __fedrise_merging(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        K = len(lsets)
        rad_info = None

        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore).to(self.device)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)
        stacked_deltawvec = self.aggwvec_tminus1 - stacked_wvec

        votedmean_dwvec, repute_info = self.sign_voted_mean(stacked_deltawvec)

        self.mom_deltawvec = (1-self.mom_beta)*votedmean_dwvec + \
                                self.mom_beta*self.mom_deltawvec
        new_wvec = self.aggwvec_tminus1 - self.mom_deltawvec.view(-1)

        # new_wvec = self.aggwvec_tminus1 - votedmean_dwvec.view(-1)
        agg_state = fedops.set_param_in_state(state_dict_struct, new_wvec,
                                               keys_to_ignore=self.vec_state_ignore)

        self.aggwvec_tminus1 = new_wvec.detach()

        info_dict = {"client_clip_weightage": rad_info, "repute_score": repute_info}


        return agg_state, info_dict

## Aliasing
FedRiseV2ΞByzantine = FedSECAΞByzantine

##==============================================================================