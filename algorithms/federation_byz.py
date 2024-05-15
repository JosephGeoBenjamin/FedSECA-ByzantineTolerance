# Ξ - chi used as underscore in Classes

import copy
import h5py

import torch
import torch.nn.functional as torch_F
import numpy as np
import random
import scipy
import pyod
import algorithms.federation_ops as fedops
import algorithms.byzantine_attacks as byz_attacks

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

def get_attack_func(method_string):
    attack_func = getattr(byz_attacks, method_string)
    return attack_func

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
        byz_clients = [int(b) for b in self.byztn_cfg["byztn_clients"]]
        if self.id in byz_clients:
            self.byz_way = get_attack_func(self.byztn_cfg["byztn_method"])(
                self.cfg, self.gmodel_init, self.device)


    #-------- Client methods ----------

    # @instancemethod #Locals calculation to send to Global L->G
    # used after One set local-rounds at each client
    def synopsize_local(self, zxs):
        """ zxs: {"model", }
        """
        lset = {}
        model = fedops.model_copier(zxs["model"])

        if self.byz_way:
            out_state = self.byz_way.modify(model.state_dict(),
                                            self.gmodel_tminus1.state_dict(),
                                            omniscience=zxs["omniscience"])
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
                    keys_to_ignore = self.vec_state_ignore)
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


##------------------------------------------------------------------------------

class NoGuardΞByzantineDecopl():
    """
    Prototype class for Decoupled FedAvg (very similar to decentralized mathematically)
        In this each client will recieve differnt set of parameters at end
        of each round, since averaging weightage for models will vary for each
        client based on closeness to different models on some metric space
    In this, simply each model is given equal weightage, this just implementation place holder

    """

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        self.aggregator_func = self.__plain_fedavg #override this to introduce methods
        print("DEFENSE: None decouple")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()


    def _init_byzantiness(self):
        byz_clients = [int(b) for b in self.byztn_cfg["byztn_clients"]]
        if self.id in byz_clients:
            self.byz_way = get_attack_func(self.byztn_cfg["byztn_method"])(
                self.cfg, self.gmodel_init, self.device)
            print("BYZ METHOD: ", self.byztn_cfg["byztn_method"])


    #-------- Client methods ----------

    # @instancemethod #Locals calculation to send to Global
    def synopsize_local(self, zxs): #used at end of local round at each client
        """ zxs: {"model", }
        Param Compression / Differential privacy or any other param modifications
        are to be carried out here
        """
        lset = {}
        model = fedops.model_copier(zxs["model"]) #after a local-rounds set

        if self.byz_way:
            out_state = self.byz_way.modify(model.state_dict(),
                                            self.gmodel_tminus1.state_dict(),
                                            omniscience=zxs["omniscience"])
        else:
            out_state = model.state_dict()

        lset["model_state"] = out_state

        return lset

    #-------- Shared methods ----------

    # @instancemethod #process global info for local use
    def desynopsize_local(self, gset, device=None, model_struct=None): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"model", }
        Decompression / local personalization of global model here
        """
        model_struct = self.gmodel_init if not model_struct else model_struct
        if not device: device = next(model_struct.parameters()).device

        ## since no compression or sketching used
        model = copy.deepcopy(model_struct)
        if not gset: return model, {}

        ## NOTE: each client can/will access its own state_dicts alone
        ## general dict notion is for easier code design
        state_cli = gset["model_states_cli"][f"client_{self.id}"]

        model.load_state_dict(state_cli, strict=True)
        model = model.to(device)

        ghatch = {}
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device)

        return model, ghatch


    #-------- Server methods ----------

    def __plain_fedavg(self, lsets):
        ## Regular
        # local_states = []
        # for ls in lsets:
        #     local_states.append(ls["model_state"])
        # agg_states = fedops.global_average_statedict(local_states)


        ##Vectorized
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        wvecs = [fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore = self.vec_state_ignore)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)

        agg_states = fedops.set_param_in_state(state_dict_struct, stacked_wvec.mean(dim=0),
                                              keys_to_ignore=self.vec_state_ignore)


        agg_states_cli = { f"client_{i}": copy.deepcopy(agg_states)
                          for i in range(len(lsets))}
        agg_states_cli.update({"client_G": copy.deepcopy(agg_states)}) #Global Model

        info_dict = {"client_weightage":[[1/len(lsets)]*len(lsets)]*len(lsets)}
        return agg_states_cli, info_dict


    # @instancemethod  #Global calculation to send to locals
    def aggregate_globally(self, lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """

        agg_states_cli, select_info = self.aggregator_func(lsets)

        ## NOTE: typically each client will have access only to its model params
        ## but returning entire dict for sake of easier code design
        gset = {"model_states_cli": agg_states_cli, "client_select": select_info}
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
                    keys_to_ignore = self.vec_state_ignore)
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
        self.cwtm_beta = self.defense_cfg.get("cwtm_beta") #B; should hold K-2B > 0
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
                    keys_to_ignore = self.vec_state_ignore)
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
                    keys_to_ignore = self.vec_state_ignore)
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
        # fully_wvecs = [fedops.get_param_from_state(l["model_state"])
        #             for l in lsets]
        # sfully_wvec = torch.vstack(fully_wvecs)

        wvecs = [fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore = self.vec_state_ignore)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)

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
                        keys_to_ignore=self.vec_state_ignore)
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
                    keys_to_ignore=self.vec_state_ignore)
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
                    keys_to_ignore=self.vec_state_ignore)
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
                    keys_to_ignore=self.vec_state_ignore)
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
                        keys_to_ignore=self.vec_state_ignore)
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


    def __ties_merging(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        K = len(lsets)

        #for l2norms
        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)
        stacked_deltawvec = self.aggwvec_tminus1 - stacked_wvec

        # for mag and sgn vectors
        magn_dwvec = torch.abs(stacked_deltawvec).view(K,-1)

        qs = magn_dwvec.quantile(self.tm_beta, dim=1).view(-1, 1)
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


##---------------------------------------------------------------------------------------



##======================================================================================


class FedRiseV2ΞByzantine(NoGuardΞByzantine):

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

        print(f"Defense: FedRISE beta-{self.mom_beta} gamma-{self.tm_gamma}")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------
    def init_stacked_wvecs(self, model):
        wvec = fedops.get_param_from_state(model.state_dict(),
                        keys_to_ignore=self.vec_state_ignore)
        self.aggwvec_tminus1 = wvec
        self.mom_deltawvec = torch.zeros_like(wvec)
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


    # ------------ Gradient Clipping -------------------------------------------

    def _locwise_grad_clamper(self, xs):
        """
        Median Clamping
        x : vectors   :shp:[K, len_parameters]
        """

        med_mag,_ = torch.median(torch.abs(xs), dim=0)

        vs = torch.clamp(xs, max=med_mag, min=-med_mag)

        return vs


    ## ----------------------- Reputation --------------------------------------

    def _torch_kendallTauA(self, a, b):
        ## tau_a = (P - Q) / (N(N-1)/2)
        ## tau_b = (P - Q) / sqrt((P + Q + T) * (P + Q + U))

        a_sgn = torch.sign(a)
        b_sgn = torch.sign(b)

        sgn_pair = a_sgn * b_sgn

        n_conc = (sgn_pair>0).sum(dim=1)
        n_disc = (sgn_pair<0).sum(dim=1)
        n = torch.prod(torch.tensor(b_sgn[0].shape)) #number of params

        # taua = (n_conc - n_disc) / torch.prod(torch.tensor(a_sgn.shape))
        taua = (n_conc - n_disc) / (n_conc+n_disc)
        # taua = (n_conc - n_disc) / (n*(n-1)/2)

        return taua


    def _reputation_score(self, sign_x):
        # mom_rep = 0.0 #fixed

        score_list = []
        for i in range(sign_x.shape[0]):
            # score = torch_F.cosine_similarity(sign_x , sign_x[i].view(1,-1))
            score = self._torch_kendallTauA(sign_x , sign_x[i].view(1,-1))
            s = torch.sign(score).mean()
            score_list.append(s)
        current_repute = torch.vstack(score_list)

        # self.prior_repute  = mom_rep*self.prior_repute + (1-mom_rep)*current_repute
        # repute = torch.clamp(self.prior_repute, min=0)

        repute = torch.clamp(current_repute, min=0)
        return repute


    ## ----------------------- Merging --------------------------------------

    def sign_voted_mean(self, dw):
        x = dw

        ## clamp the max grads
        x = self._locwise_grad_clamper(x)

        ## mag and sgn vectors -> for ties
        magn_x = torch.abs(x)

        ql = magn_x.quantile(self.tm_gamma, dim=1)
        ql = ql.view(-1, 1)
        x[magn_x<ql] = 0.0

        sign_x = torch.sign(x)

        ## vote with repute
        repute = self._reputation_score(sign_x)

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
                    keys_to_ignore=self.vec_state_ignore)
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

        self.aggwvec_tminus1 = new_wvec.clone()

        info_dict = {"client_clip_weightage": rad_info, "repute_score": repute_info}

        return agg_state, info_dict