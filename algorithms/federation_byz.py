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

        self.aggregator_func = self.__plain_fedavg #override this to introduce methods
        print("DEFENSE: None")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()


    def _init_byzantiness(self):
        byz_clients = [int(b) for b in self.byztn_cfg["byztn_clients"]]
        if self.id in byz_clients:
            self.byz_way = get_attack_func(self.byztn_cfg["byztn_method"])(self.cfg, self.gmodel_init)
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
                                            self.gmodel_tminus1.state_dict())
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
        local_states = []
        for ls in lsets:
            local_states.append(ls["model_state"])
        agg_state = fedops.global_average_statedict(local_states, device="cpu")

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
    In this simply each model is given equal weightage, this just implementation place holder

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

        self.aggregator_func = self.__plain_fedavg #override this to introduce methods
        print("DEFENSE: None decouple")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()


    def _init_byzantiness(self):
        byz_clients = [int(b) for b in self.byztn_cfg["byztn_clients"]]
        if self.id in byz_clients:
            self.byz_way = get_attack_func(self.byztn_cfg["byztn_method"])(self.cfg, self.gmodel_init)
            print("BYZ METHOD: ", self.byztn_cfg["byztn_method"])


    #-------- Client methods ----------

    # @instancemethod #Locals calculation to send to Global
    def synopsize_local(self, zxs): #used at end of local round at each client
        """ zxs: {"model", }
        """
        lset = {}
        model = fedops.model_copier(zxs["model"]) #after a local-rounds set

        if self.byz_way:
            out_state = self.byz_way.modify(model.state_dict(),
                                            self.gmodel_tminus1.state_dict())
        else:
            out_state = model.state_dict()

        lset["model_state"] = out_state

        return lset

    #-------- Shared methods ----------

    # @instancemethod #process global info for local use
    def desynopsize_local(self, gset, device=None, model_struct=None): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"model", }
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
        local_states = []
        for ls in lsets:
            local_states.append(ls["model_state"])

        agg_states = fedops.global_average_statedict(local_states)

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

## TODO: fix model state reading directly

class KrumΞByzantine(NoGuardΞByzantine):
    """
    Work: https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html

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

        self.aggregator_func = self.__krum_aggregation
        self.krum_m  = int(self.defense_cfg["multikrum_m"]) # M
        self.num_byz_b  = self.defense_cfg["num_assumed_byz"] # B; should hold 2b+2 < n

        self.vec_state_ignore = ["num_batches_tracked"]
        if not ( (2*self.num_byz_b+2) < self.num_client_k):
            print(f"WARNING!!!..... `2b+2 < n` doesn't hold, 2*{self.num_byz_b}+2 < {self.num_client_k} ")
        print("Defense: Krum")


        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()



    #-------- Server methods ----------

    def __krum_aggregation(self, lsets):
        fully_wvecs = [fedops.get_param_from_state(l["model_state"])
                    for l in lsets]

        wvecs = [fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore = self.vec_state_ignore)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)

        num_neigbour = self.num_client_k - self.num_byz_b - 2

        all_dist = []
        for v in wvecs:
            all_dist.append(torch.norm(stacked_wvec-v, dim=1))

        neighbor_dist_sum = []
        for dist in all_dist:
            vals, idxs = torch.topk(dist, k=num_neigbour+1, #since top will include self distance i.e zero
                       largest=False)
            neighbor_dist_sum.append(vals.sum())

        mvals, midxs = torch.topk(torch.hstack(neighbor_dist_sum),
                                  k=self.krum_m, largest=False)

        final_wvec = torch.zeros_like(fully_wvecs[0])
        for mi in midxs.tolist():
            final_wvec +=fully_wvecs[mi]
        final_wvec /= len(midxs)

        agg_state = fedops.set_param_in_state(lsets[0]["model"].state_dict(), final_wvec)


        info_dict = {"client_weightage": [ 1/len(midxs) if i in midxs else 0
                                        for i in range(len(lsets))]
                    }
        return agg_state, info_dict


##------------------------------------------------------------------------------

from pyod.models.copod import COPOD

class CopodDosΞByzantine(NoGuardΞByzantine):
    """
    Work: https://arxiv.org/abs/2207.10804
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

        self.aggregator_func = self.__dos_aggregation
        self.cpd_l2 = COPOD()
        self.cpd_cs = COPOD()

        self.vec_state_ignore = ["num_batches_tracked"]

        print("Defense: Copod-DOS")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()



    #-------- Server methods ----------

    def __dos_aggregation(self, lsets):
        fully_wvecs = [fedops.get_param_from_state(l["model_state"])
                    for l in lsets]
        sfully_wvec = torch.vstack(fully_wvecs)

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
        cweighed_wvec = cweigh.to(self.device) * sfully_wvec  # s1*[v1] \ s2*[v2] \ s3*v3 ...

        final_wvec = torch.sum(cweighed_wvec, axis = 0)

        agg_state = fedops.set_param_in_state(lsets[0]["model_state"], final_wvec)

        info_dict = {"client_weightage":cweigh.flatten().tolist()}
        return agg_state, info_dict


##------------------------------------------------------------------------------
##==============================================================================


## Decoupled weightage based on Sinkhorn distance

class WeighOmegaSKDHΞByzantineDecopl(NoGuardΞByzantineDecopl):

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

        self.aggregator_func = self.__dataweightage_aggregation_decopld

        with h5py.File(self.defense_cfg["datasummary"], 'r') as hdf5_file:
            data_dist = hdf5_file["dist_matrix"][()]
            data_dist = data_dist[:-1, :-1] # ignore all distances

        self.client_weightage = self._preset_weightage(data_dist)

        print("Defense: Decoupled WeighOmega-SKHD")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------

    def _preset_weightage(self, data_dist):
        K = data_dist.shape[0]

        ## FOR ISIC dataset
        # matx = torch.tensor([0.53,0.17,0.14,0.10,0.04,0.02]*K).view(K,K)

        matx = torch.ones((K,K)) * (1/K)

        return matx

    def _distance_naive_softmin_weightage(self, data_dist):
        data_dist = (data_dist + data_dist.T) / 2

        data_dist = torch.tensor(data_dist)

        normed_dist = (data_dist - data_dist.min(dim=1, keepdim=True)[0]) /  \
                    (data_dist.max(dim=1, keepdim=True)[0] - data_dist.min(dim=1, keepdim=True)[0])

        weightage_matrix = torch_F.softmin(torch.tensor(normed_dist), dim=1)
        return weightage_matrix

    def _mena_soft_weightage(self, data_dist):
        data_dist = (data_dist + data_dist.T) / 2
        data_dist = torch.tensor(data_dist)

        K = data_dist.shape[0] #clients

        rmean_dist = torch.mean(data_dist, dim=0)

        all_dist =  (data_dist +
            rmean_dist * torch.eye(K, dtype=float).to_dense())

        rel_dist = all_dist / rmean_dist.view(-1,1)
        rel_dist = torch.clamp(rel_dist, min=1.0)

        weightage_matrix = torch_F.softmin(rel_dist)

        return weightage_matrix


    def _boltzman_factor_weightage(self, data_dist):
        """ Follows Boltzman Distribution paradigm
        data_dist: numpy arr
        return : torch.tensor
        """
        data_dist = (data_dist + data_dist.T) / 2
        # data_dist = data_dist.T
        K = data_dist.shape[0]

        data_dist = torch.tensor(data_dist)

        ## 90th - 1.282 | 95th - 1.645 | 99th - 2.326 | 75th - 0.674
        sigz = 1.645

        ## COMPUTE Temperature
        non_diag = torch_remove_diagonal(data_dist)
        dist_mu = torch.mean(non_diag)
        sigma = torch.std(non_diag)

        dist_max, dist_min = dist_mu+sigz*sigma, dist_mu-sigz*sigma
        tempK = (dist_max - dist_min) / (np.log(1/6) - np.log(5/6))

        ### COMPUTE beta_ij
        softin_beta = non_diag / tempK
        beta_ij = torch.softmax(softin_beta, dim=1)

        ## randomize compute betas along row
        # perm = torch.randperm(beta_ij.shape[1])
        # beta_ij = beta_ij[:, perm]

        ## constant betas
        # beta_ij = beta_ij*0 + (1/5)

        beta_ij = torch_insert_diagonal(beta_ij)

        ### COMPUTE alpha_ii
        # drowmean = (data_dist.sum(dim=0)/(data_dist.shape[0]-1))
        # softin_alpha = drowmean / tempK
        # alpha_ii = torch.softmax(softin_alpha, dim=0)
        # alpha_ii = torch.max(beta_ij, dim=1)[0]

        ## constant alpha
        alpha_ii = 1.5/ data_dist.shape[0] #approx 0.25 for isic

        # omega_ij full
        weightage_matrix = ((1-alpha_ii) * beta_ij +
                        alpha_ii * torch.eye(data_dist.shape[0], dtype=float).to_dense())

        return weightage_matrix

    #-------- Server methods ----------

    def __dataweightage_aggregation_decopld(self, lsets):
        fully_wvecs = [fedops.get_param_from_state(l["model_state"])
                    for l in lsets]
        sfully_wvec = torch.vstack(fully_wvecs)
        out_ref_wvec = torch.zeros_like(sfully_wvec)

        agg_states_cli = {}
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])
        for i in range(len(lsets)):
            cweigh = self.client_weightage[i,:].view(-1, 1)
            cweighed_wvec = cweigh.to(self.device) * sfully_wvec  # s1*[v1] \ s2*[v2] \ s3*v3 ...
            cli_wvec = torch.sum(cweighed_wvec, axis = 0)
            agg_states = fedops.set_param_in_state(state_dict_struct, cli_wvec)
            agg_states_cli.update({f"client_{i}": copy.deepcopy(agg_states)})
            out_ref_wvec[i, :] = cli_wvec

        agg_states = fedops.set_param_in_state(state_dict_struct, out_ref_wvec.mean(dim=0))
        agg_states_cli.update({"client_G": copy.deepcopy(agg_states)})

        info_dict = {"client_weightage":self.client_weightage.tolist()}
        return agg_states_cli, info_dict


##==============================================================================


class TauThetaLambdaSKDHΞByzantineDecopl(NoGuardΞByzantineDecopl):

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

        self.aggregator_func = self.__dynamic_Tau_Theta_Lambda_aggr_decopld

        with h5py.File(self.defense_cfg["datasummary"], 'r') as hdf5_file:
            data_dist = hdf5_file["dist_matrix"][()]
            data_dist = data_dist[:-1, :-1] # ignore all distances

        self.client_clip_factor = self._get_lmbda_from_dist(data_dist)

        ##
        self.vec_state_ignore = ["num_batches_tracked"] # critical for l2norms since this skews it
        if self.id == "G": #large tensors, so why waste mem
            self.init_stacked_wvecs()

        print("Defense: Decoupled Clipping Tau-SKHD")


        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------

    def init_stacked_wvecs(self):
        wvec = fedops.get_param_from_state(self.gmodel_init.state_dict(),
                        keys_to_ignore=self.vec_state_ignore)
        self.wvec_init = wvec
        self.aggwvec_tminus1 = torch.vstack( [wvec]*self.num_client_k )
        self.wvec_tminus1    = self.aggwvec_tminus1.clone()
        fully_wvec =  fedops.get_param_from_state(self.gmodel_init.state_dict())
        self.fully_aggwvec_tminus1 = torch.vstack( [fully_wvec]*self.num_client_k )


    def _get_constant_tau(self, data_dist):
        K = data_dist.shape[0]
        tau_val = 1.0
        matx = torch.ones((K,K)) * tau_val
        return matx


    def _get_lmbda_from_dist(self, data_dist):
        data_dist = (data_dist + data_dist.T) / 2
        data_dist = torch.tensor(data_dist)
        K = data_dist.shape[0] #clients

        # rmean_dist = torch.sum(data_dist, dim=0) / K
        # all_dist = data_dist + (rmean_dist * torch.eye(K, dtype=float).to_dense())

        own_dist = torch.diag(data_dist)

        lambda_dist = torch.div(data_dist, own_dist.view(-1, 1))
        lambda_dist = 1 + torch.abs(1 - lambda_dist)

        return lambda_dist


    #-------- Server methods ----------
    def safe_divide(self, nu, de, fill=1.0):
        res = torch.full_like(de, fill_value=fill)
        mask = (de != 0.0)

        if (nu.shape == mask.shape): nu_ = nu[mask]
        elif (sum(nu.shape) == 1):   nu_ = nu
        else: raise Exception(f"Incompatible shapes {de.shape}, {nu.shape}")

        res[mask] = torch.div(nu_, de[mask])
        return res


    def __dynamic_Tau_Theta_Lambda_aggr_decopld(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])

        #for fedavging with bn_batches_tracked; thanks to Pytorch default models for complicating life
        fully_wvecs = [fedops.get_param_from_state(l["model_state"])
                    for l in lsets]
        sfully_wvec = torch.vstack(fully_wvecs)
        outfully_aggwvec = torch.zeros_like(sfully_wvec)
        sfully_aggwvec_tminus1 = self.fully_aggwvec_tminus1

        #for l2norms
        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)
        outref_aggwvec = torch.zeros_like(stacked_wvec)

        stacked_wvec_tminus1    = self.wvec_tminus1
        stacked_aggwvec_tminus1 = self.aggwvec_tminus1

        ##
        agg_states_cli = {}
        lmbda_dist = self.client_clip_factor.to(self.device)

        print(torch.norm(stacked_wvec_tminus1 - stacked_wvec[0], dim=1).view(-1, 1))

        ## Tau Computes
        tau_rad = torch.norm(self.wvec_init - stacked_wvec, dim=1).view(-1, 1)    #---> [5]
        ## Theta Computes
        cos_theta = torch_F.cosine_similarity(self.wvec_init, stacked_wvec, dim=1).view(-1, 1)

        sector_scales = []; rad_scales = []; cos_scales = []
        for i in range(stacked_wvec.shape[0]):
            taui = tau_rad[i]
            ccden = torch.norm(stacked_wvec - stacked_wvec[i], dim=1).view(-1,1)
            taui_by_ccden = self.safe_divide(taui, ccden)
            rad_comp = torch.minimum(torch.tensor(1), taui_by_ccden).view(-1,1)

            thetai = cos_theta[i]
            cosbas = torch_F.cosine_similarity(stacked_wvec, stacked_wvec[i], dim=1).view(-1,1)
            cosbas_x_thetai = thetai*100*(cosbas-thetai)
            cos_comp = torch_F.sigmoid(cosbas_x_thetai)

            scale_sec = rad_comp * cos_comp #* lmbda_dist[i].view(-1,1)
            scale_sec = torch.clamp(scale_sec, max=1, min=0)
            # print("Scale Shape", scale_sec.shape)

            ## start core
            clipped_fully_deltawvec = scale_sec * (sfully_wvec - sfully_aggwvec_tminus1) # s1*[v1] \ s2*[v2] \ s3*v3 ...
            clipped_fully_aggdeltawvec = \
                torch.sum(clipped_fully_deltawvec, axis = 0) / torch.sum(scale_sec)

            cli_fully_wvec = sfully_aggwvec_tminus1[i] + clipped_fully_aggdeltawvec
            outfully_aggwvec[i, :] = cli_fully_wvec

            sector_scales.append(scale_sec.flatten().tolist())
            rad_scales.append(rad_comp.flatten().tolist())
            cos_scales.append(cos_comp.flatten().tolist())

            agg_states = fedops.set_param_in_state(state_dict_struct, cli_fully_wvec)
            agg_states_cli.update({f"client_{i}": copy.deepcopy(agg_states)})
            ## end core

            outref_aggwvec[i, :] = fedops.get_param_from_state(agg_states,
                                        keys_to_ignore=self.vec_state_ignore)


        agg_states = fedops.set_param_in_state(state_dict_struct, outfully_aggwvec.mean(dim=0))
        agg_states_cli.update({"client_G": copy.deepcopy(agg_states)})

        self.wvec_tminus1    = stacked_wvec.clone()
        self.aggwvec_tminus1 = outref_aggwvec.clone()
        self.fully_aggwvec_tminus1 = outfully_aggwvec.clone()

        info_dict = {"client_clip_weightage":sector_scales,
                     "radius_component": rad_scales,
                     "cosine_component": cos_scales}

        return agg_states_cli, info_dict