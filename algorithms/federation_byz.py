# Ξ - chi used as underscore in Classes

import copy
import h5py

import torch
import torch.nn.functional as torch_F
import numpy as np
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
            print("BYZ METHOD: ", self.byztn_cfg["byztn_method"])


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
            all_dist.append(torch.norm(stacked_wvec-v, dim=1).view(K,1))

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

            # median_wvec = torch.tensor(0.0) ## to disable median centring
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
        l2_dists = torch.vstack(l2_dists).cpu()
        cs_dists = torch.vstack(cs_dists).cpu()

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
        with torch.no_grad():
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

class ClippingBucketingΞByzantine(NoGuardΞByzantine): # Attempt 2

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

        self.aggregator_func = self.__custom_aggregate

        ##
        self.vec_state_ignore = ["num_batches_tracked"] # critical for l2norms since this skews it
        if self.id == "G": #large tensors, so why waste mem
            self.init_stacked_wvecs(model)

        print("Defense: Clipping Bucketing")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------

    def init_stacked_wvecs(self, model):
        wvec = fedops.get_param_from_state(model.state_dict(),
                        keys_to_ignore=self.vec_state_ignore)
        self.wvec_init = wvec.clone()
        self.aggwvec_tminus1 = wvec.clone()


    def get_tau(self):
        beta = 0.9
        return torch.tensor(500).view(1).to(self.device)


    #-------- Server methods ----------
    def safe_divide(self, nu, de, fill=1.0):
        res = torch.full_like(de, fill_value=fill)
        mask = (de != 0.0)

        if (nu.shape == mask.shape): nu_ = nu[mask]
        elif (sum(nu.shape) == 1):   nu_ = nu
        else: raise Exception(f"Incompatible shapes {de.shape}, {nu.shape}")

        res[mask] = torch.div(nu_, de[mask])
        return res

    def __custom_aggregate(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])

        #for l2norms
        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)
        stacked_deltawvec = torch.zeros_like(stacked_wvec)

        K = len(lsets)

        tau = self.get_tau()

        momentum = self.aggwvec_tminus1.clone()

        ## Clipping
        for m in range(self.clip_iters):
            stacked_gdelta = torch.norm(stacked_wvec-momentum, dim=1).view(-1, 1) #

            tau_by_ccden = self.safe_divide(tau, stacked_gdelta, fill=0.0) ## just setting clipping radius for zero norm
            rad_comp = torch.minimum(torch.tensor(1), tau_by_ccden).view(-1,1)

            clipped_deltawvec = rad_comp * (stacked_wvec - momentum) # s1*[v1] \ s2*[v2] \ s3*v3 ...
            clipped_aggdelta = torch.mean(clipped_deltawvec, dim = 0)

            momentum = momentum + clipped_aggdelta


        agg_state = fedops.set_param_in_state(state_dict_struct, momentum,
                                               keys_to_ignore=self.vec_state_ignore)

        self.aggwvec_tminus1 = momentum.clone()

        rad_scales = rad_comp.flatten().tolist()
        info_dict = {"client_clip_weightage":rad_scales}

        return agg_state, info_dict


##==============================================================================

##==============================================================================


class NewNewTauΞByzantine(NoGuardΞByzantine): # Attempt 2

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
        self.vec_state_ignore = ["num_batches_tracked"] # critical for l2norms since this skews it


        # self.clip_iters = int(self.defense_cfg.get("clip_iters")) # if zero no clipping will happen
        # if self.clip_iters==0: print("Clipping disabled since clip iters is 0")

        self.aggregator_func = self.__custom_two_aggregate

        ##
        if self.id == "G": #large tensors, so why waste mem
            self.init_stacked_wvecs(model)

        print("Defense: LOTR Two Towers")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------

    def init_stacked_wvecs(self, model):
        wvec = fedops.get_param_from_state(model.state_dict(),
                        keys_to_ignore=self.vec_state_ignore)
        self.wvec_init = wvec.clone()
        self.aggmvec_tminus1 = wvec.clone()
        self.stacked_mveci_tminus1 = torch.vstack( [wvec]*self.num_client_k )


    def get_fixed_tau(self):
        beta = 0.9
        return torch.tensor(500).view(1).to(self.device)


    #-------- Server methods ----------
    def safe_divide(self, nu, de, fill=1.0):
        res = torch.full_like(de, fill_value=fill)
        mask = (de < 1e-12)

        if (nu.shape == mask.shape): nu_ = nu[mask]
        elif (sum(nu.shape) == 1):   nu_ = nu
        else: raise Exception(f"Incompatible shapes {de.shape}, {nu.shape}")

        res[mask] = torch.div(nu_, de[mask])
        return res

    def __custom_one_aggregate(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        K = len(lsets)

        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)

        aggmvec_tminus1 = self.aggmvec_tminus1.clone().view(1, -1)
        stacked_mveci_tminus1 = self.stacked_mveci_tminus1

        tau = self.get_fixed_tau()

        ## Clientwise reference Clipping
        mi_scales = []
        outk_mveci = torch.zeros_like(stacked_wvec)
        for i in range(stacked_wvec.shape[0]):

            stacked_inorm = torch.norm(stacked_wvec-stacked_mveci_tminus1[i], dim=1).view(-1,1)
            stacked_icosr = torch.acos(torch_F.cosine_similarity(  # radians
                            stacked_wvec, stacked_mveci_tminus1[i], dim=1)).view(-1,1)
            ccdeni = (stacked_inorm + stacked_icosr) / 2
            taui_by_ccdeni = self.safe_divide(tau, ccdeni, fill=1.0) # this will return 1 for taui by ccdenii
            scale_seci = torch.minimum(torch.tensor(1), taui_by_ccdeni).view(-1,1)

            clipped_delta_mi = scale_seci * (stacked_wvec - stacked_mveci_tminus1[i]) # s1*[v1] \ s2*[v2] \ s3*v3 ...
            clipped_delta_mi[i] = 0 # remove i-th client update from momentum_i
            clipped_delta_mi = torch.sum(clipped_delta_mi, dim = 0) / (K-1)

            mveci = stacked_mveci_tminus1[i] + clipped_delta_mi
            outk_mveci[i, :] = mveci
            mi_scales.append(scale_seci.flatten().tolist())

        ## Global reference clipping
        stacked_gnorm = torch.norm(outk_mveci-aggmvec_tminus1, dim=1).view(-1, 1) #
        stacked_gcosr = torch.acos(torch_F.cosine_similarity(  # radians
                            outk_mveci, aggmvec_tminus1, dim=1)).view(-1,1)
        ccdeng = (stacked_gnorm + stacked_gcosr) / 2
        taug_by_ccdeng = self.safe_divide(tau, ccdeng, fill=0.0) ## just setting clipping radius for zero norm
        scale_secg = torch.minimum(torch.tensor(1), taug_by_ccdeng).view(-1,1)

        clipped_delta_g = scale_secg * (outk_mveci - aggmvec_tminus1) # s1*[v1] \ s2*[v2] \ s3*v3 ...
        clipped_delta_g = torch.mean(clipped_delta_g, dim = 0)

        aggmvec = aggmvec_tminus1 + clipped_delta_g

        agg_state = fedops.set_param_in_state(state_dict_struct, aggmvec.view(-1),
                                                keys_to_ignore=self.vec_state_ignore)
        self.aggwvec_tminus1 = aggmvec.clone()
        self.stacked_mveci_tminus1 = outk_mveci.clone()

        g_scales = scale_secg.flatten().tolist()

        info_dict = {"client_clip_weightage":mi_scales,
                     "global_clip_weightage":g_scales}

        return agg_state, info_dict


    def __custom_two_aggregate(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        K = len(lsets)

        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)

        aggmvec_tminus1 = self.aggmvec_tminus1.clone().view(1, -1)
        stacked_mveci_tminus1 = self.stacked_mveci_tminus1

        torch_pi = torch.tensor(np.pi)

        ## Clientwise reference Clipping
        mi_scales = []
        mitheta_scales = []
        outk_mveci = torch.zeros_like(stacked_wvec)
        for i in range(stacked_wvec.shape[0]):

            inorm = torch.norm(stacked_wvec[i]-aggmvec_tminus1).view(1)
            taui = inorm

            stacked_inorm = torch.norm(stacked_wvec-aggmvec_tminus1, dim=1).view(-1,1)
            ccdeni = stacked_inorm

            taui_by_ccdeni = self.safe_divide(taui, ccdeni, fill=1.0) # this will return 1 for taui by ccdenii
            scale_seci = torch.minimum(torch.tensor(1), taui_by_ccdeni).view(-1,1)

            thcosi =  torch.acos(torch_F.cosine_similarity(  # radians
                            stacked_wvec, aggmvec_tminus1.view(1,-1))).view(-1,1)
            # thcosi_rel = (thcosi[i] / thcosi)
            # scale_thetai = torch.exp(2*torch_pi*(thcosi_rel-1))
            # scale_thetai = torch.minimum(torch.tensor(1), scale_thetai).view(-1,1)

            thcosi_std = thcosi.std()
            if   (thcosi.median() < thcosi.mean()): thcosi_std = thcosi[thcosi<thcosi.median()].std()
            elif (thcosi.median() > thcosi.mean()): thcosi_std = thcosi[thcosi>thcosi.median()].std()
            thcosi_std = thcosi_std.nan_to_num(nan=1)
            scale_thetai = torch.exp(-0.5*torch.square((thcosi-thcosi.median())/thcosi_std))

            # breakpoint()
            scale_thetai = 1.0 - torch.isclose(scale_thetai, torch.zeros_like(scale_thetai)).float()

            # clipped_delta_mi = scale_seci * scale_thetai *(stacked_wvec - aggmvec_tminus1) # s1*[v1] \ s2*[v2] \ s3*v3 ...
            # clipped_delta_mi[i, :] = 0 # remove i-th client update from momentum_i
            # clipped_delta_mi = torch.sum(clipped_delta_mi, dim = 0) / (K-1)
            # mveci = aggmvec_tminus1 + clipped_delta_mi

            rescaled_wvec = scale_seci * scale_thetai * stacked_wvec
            rescaled_wvec[i, :] = 0
            mveci = torch.sum(rescaled_wvec, dim=0) / (K-1)

            outk_mveci[i, :] = mveci
            mi_scales.append(scale_seci.flatten().tolist())
            mitheta_scales.append(scale_thetai.flatten().tolist())

        ## Global reference clipping

        # gnorm = torch.norm(stacked_wvec-aggmvec_tminus1, dim=1).view(-1,1)
        # gcosr =  torch.acos(torch_F.cosine_similarity(  # radians
        #                 stacked_wvec, aggmvec_tminus1, dim=1)).view(-1,1)
        # taug = (gnorm+gcosr) / 2

        # stacked_gnorm = torch.norm(outk_mveci-aggmvec_tminus1, dim=1).view(-1, 1) #
        # stacked_gcosr = torch.acos(torch_F.cosine_similarity(  # radians
        #                     outk_mveci, aggmvec_tminus1, dim=1)).view(-1,1)
        # ccdeng = (stacked_gnorm + stacked_gcosr) / 2

        # taug_by_ccdeng = self.safe_divide(taug, ccdeng, fill=0.0) ## just setting clipping radius for zero norm
        # scale_secg = torch.minimum(torch.tensor(1), taug_by_ccdeng).view(-1,1)

        semi_aggmvec = torch.mean(outk_mveci, dim=0)
        recov_stacked_wvec = semi_aggmvec - outk_mveci

        gradients_recov = recov_stacked_wvec - aggmvec_tminus1
        gradients_mveci = outk_mveci - aggmvec_tminus1
        gradients_wvec  = stacked_wvec-aggmvec_tminus1

        thcosg =  torch.acos(torch_F.cosine_similarity(  # radians
                        gradients_mveci, gradients_recov)).view(-1,1)

        # scale_thetag = torch.exp(-2*torch_pi*(thcosg-torch_pi/2))
        # scale_thetag = torch.minimum(torch.tensor(1), scale_thetag).view(-1,1)

        thcosg_std = thcosg.std()
        if   (thcosg.median() < thcosg.mean()): thcosg_std = thcosg[thcosg<thcosg.median()].std()
        elif (thcosg.median() > thcosg.mean()): thcosg_std = thcosg[thcosg>thcosg.median()].std()
        thcosg_std = thcosg_std.nan_to_num(nan=1)
        scale_thetag = torch.exp(-0.5*torch.square((thcosg-thcosg.median())/thcosg_std))
        scale_thetag = 1.0 - torch.isclose(scale_thetag,  torch.zeros_like(scale_thetag)).float()


        print(thcosg.view(1,-1), thcosg_std.view(1,-1))
        print(scale_thetag)

        test_ang = torch.acos(torch_F.cosine_similarity(  # radians
                        stacked_wvec, aggmvec_tminus1)).view(-1,1)

        print("TestAng", test_ang)

        # breakpoint()

        # clipped_delta_g = scale_thetag * gradients_recov # s1*[v1] \ s2*[v2] \ s3*v3 ...
        # clipped_delta_g = torch.sum(clipped_delta_g, dim = 0) / torch.sum(scale_thetag)
        # aggmvec = aggmvec_tminus1 + clipped_delta_g

        aggmvec = (scale_thetag * outk_mveci).sum(dim=0) / torch.sum(scale_thetag)

        agg_state = fedops.set_param_in_state(state_dict_struct, aggmvec.view(-1),
                                                keys_to_ignore=self.vec_state_ignore)
        self.aggwvec_tminus1 = aggmvec.clone()
        self.stacked_mveci_tminus1 = outk_mveci.clone()

        g_scales = scale_thetag.flatten().tolist()

        info_dict = {"client_clip_weightage" :mi_scales,
                     "client_theta_weightage":mitheta_scales,
                     "global_clip_weightage" :g_scales}

        return agg_state, info_dict