# Ξ - chi used as underscore in Classes

import copy
import torch
import torch.nn.functional as torch_F
import scipy
import pyod
import algorithms.federation_ops as fedops


##==============================================================================
## COMMONS



##==============================================================================
## ATTACKS


class RandomizedζAttack():
    def __init__(self, cfg, model ):
        # self.model_0th = copy.deepcopy(model)
        self.byz_cfg = cfg.byztn_cfg

    def modify(self, model_state):
        weight_vec = fedops.get_param_from_state(model_state)
        weight_vec[:] = torch.rand(len(weight_vec))
        out_state = fedops.set_param_in_state(model_state, weight_vec)

        return out_state


class AffineζAttack():
    def __init__(self, cfg, model ):
        # self.model_0th = copy.deepcopy(model)
        self.byz_cfg = cfg.byztn_cfg

        self.scaler = cfg.byztn_cfg["scale"]

    def modify(self, model_state):
        weight_vec = fedops.get_param_from_state(model_state)
        weight_vec[:]= self.scaler * weight_vec
        out_state = fedops.set_param_in_state(model_state, weight_vec)

        return out_state


class NaifttCraftedζAttack():
    """ https://github.com/Naiftt/SPAFD/ """

    def __init__(self, cfg, model ):
        self.byz_cfg = cfg.byztn_cfg
        self.lmbd = cfg.byztn_cfg["lambda"]

        self.wvec_tminus_1 = fedops.get_param_from_state(model.state_dict())


    def modify(self, model_state):
        wvec_current = fedops.get_param_from_state(model_state)

        S = (wvec_current > self.wvec_tminus_1).long()
        S[S==0] = -1
        new_wvec =  wvec_current - (self.lmbd*S)

        out_state = fedops.set_param_in_state(model_state, new_wvec)
        self.wvec_tminus_1 = wvec_current.clone()

        return out_state



##==============================================================================
## Fed Protocol -- all are State Dict based aggregation

# FedAggregator = fedops.SimpleStateΞFedAvg

## **************************************


class NoGuardΞByzantine():

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.model_0th = copy.deepcopy(model)
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg

        self.aggregator_func = self.__plain_fedavg #override this to introduce methods
        print("DEFENSE: None")

        if len(self.byztn_cfg) != 0:
            byz_clients = [int(b) for b in self.byztn_cfg["byztn_clients"]]
            if id in byz_clients:
                self.byz_way = globals()[self.byztn_cfg["byztn_method"]](cfg, model)
                print("BYZ METHOD: ", self.byztn_cfg["byztn_method"])


    #-------- Client methods ----------

    # @instancemethod #Locals calculation to send to Global
    def synopsize_local(self, zxs): #used at end of local round at each client
        """ zxs: {"model", }
        """
        lset = {}
        model = fedops.model_copier(zxs["model"])

        if self.byz_way:
            out_state = self.byz_way.modify(model.state_dict())
        else:
            out_state = model.state_dict()

        model.load_state_dict(out_state)
        lset["model"] = model

        return lset

    #-------- Shared methods ----------

    # @instancemethod #process global info for local use
    def desynopsize_local(self, gset, device=None, model_struct=None): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"model", }
        """
        model_struct = self.model_0th if not model_struct else model_struct
        if not device: device = next(model_struct.parameters()).device

        ## since no compression or sketching used
        model = copy.deepcopy(model_struct)
        if not gset: return model, {}

        model.load_state_dict(copy.deepcopy(gset["model"].state_dict()), strict=True)
        model = model.to(device)
        ghatch = {}

        return model, ghatch


    #-------- Server methods ----------

    def __plain_fedavg(self, lsets):
        local_states = []
        for ls in lsets:
            local_states.append(ls["model"].state_dict())
        agg_states = fedops.global_average_statedict(local_states)

        info_dict = {"client_weights":[1/len(lsets)]*len(lsets)}
        return agg_states, info_dict


    # @instancemethod  #Global calculation to send to locals
    def aggregate_globally(self, lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        if not device: device = next(self.model_0th.parameters()).device
        for ls in lsets: ls["model"].to(device) #for nn.module .cuda is both inplace and assignable

        agg_model = fedops.model_copier(lsets[0]["model"])

        agg_states, select_info = self.aggregator_func(lsets)

        agg_model.load_state_dict(agg_states)

        gset = {"model": agg_model, "client_select": select_info}
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
        self.model_0th = copy.deepcopy(model)
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg

        self.aggregator_func = self.__plain_fedavg #override this to introduce methods
        print("DEFENSE: None")

        if len(self.byztn_cfg) != 0:
            byz_clients = [int(b) for b in self.byztn_cfg["byztn_clients"]]
            if id in byz_clients:
                self.byz_way = globals()[self.byztn_cfg["byztn_method"]](cfg, model)
                print("BYZ METHOD: ", self.byztn_cfg["byztn_method"])


    #-------- Client methods ----------

    # @instancemethod #Locals calculation to send to Global
    def synopsize_local(self, zxs): #used at end of local round at each client
        """ zxs: {"model", }
        """
        lset = {}
        model = fedops.model_copier(zxs["model"])

        if self.byz_way:
            out_state = self.byz_way.modify(model.state_dict())
        else:
            out_state = model.state_dict()

        model.load_state_dict(out_state)
        lset["model"] = model

        return lset

    #-------- Shared methods ----------

    # @instancemethod #process global info for local use
    def desynopsize_local(self, gset, device=None, model_struct=None): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"model", }
        """
        model_struct = self.model_0th if not model_struct else model_struct
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

        return model, ghatch


    #-------- Server methods ----------

    def __plain_fedavg(self, lsets):
        local_states = []
        for ls in lsets:
            local_states.append(ls["model"].state_dict())
        agg_states = fedops.global_average_statedict(local_states)

        agg_states_cli = { f"client_{i}": copy.deepcopy(agg_states)
                          for i in range(len(lsets))}
        agg_states_cli.update({"client_G": copy.deepcopy(agg_states)}) #Global Model

        info_dict = {"client_weights":[[1/len(lsets)]*len(lsets)]*len(lsets)}
        return agg_states_cli, info_dict


    # @instancemethod  #Global calculation to send to locals
    def aggregate_globally(self, lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        if not device: device = next(self.model_0th.parameters()).device
        for ls in lsets: ls["model"].to(device) #for nn.module .cuda is both inplace and assignable

        agg_states_cli, select_info = self.aggregator_func(lsets)

        ## NOTE: typically each client will have access only to its model params
        ## but returning entire dict for sake of easier code design
        gset = {"model_states_cli": agg_states_cli, "client_select": select_info}
        return gset


##==============================================================================



class KrumΞByzantine(NoGuardΞByzantine):
    """
    Work: https://papers.nips.cc/paper_files/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html

    """
    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.num_client_k = int(cfg.data_centers_count) # K
        self.model_0th = copy.deepcopy(model)
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg

        self.aggregator_func = self.__krum_aggregation
        self.krum_m  = int(self.defense_cfg["multikrum_m"]) # M
        self.num_byz_b  = self.defense_cfg["num_assumed_byz"] # B; should hold 2b+2 < n

        if not ( (2*self.num_byz_b+2) < self.num_client_k):
            print(f"WARNING!!!..... `2b+2 < n` doesn't hold, 2*{self.num_byz_b}+2 < {self.num_client_k} ")
        print("Defense: Krum")


        if len(self.byztn_cfg) != 0:
            byz_clients = [int(b) for b in self.byztn_cfg["byztn_clients"]]
            if id in byz_clients:
                self.byz_way = globals()[self.byztn_cfg["byztn_method"]](cfg, model)
                print("Byz Method", self.byztn_cfg["byztn_method"])



    #-------- Server methods ----------

    def __krum_aggregation(self, lsets):
        wvecs = [fedops.get_param_from_state(l["model"].state_dict()) for l in lsets]
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

        final_wvec = torch.zeros_like(wvecs[0])
        for mi in midxs.tolist():
            final_wvec +=wvecs[mi]
        final_wvec /= len(midxs)

        agg_states = fedops.set_param_in_state(lsets[0]["model"].state_dict(), final_wvec)


        info_dict = {"client_weights": [ 1/len(midxs) if i in midxs else 0
                                        for i in range(len(lsets))]
                    }
        return agg_states, info_dict


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
        self.num_client_k = int(cfg.data_centers_count) # K
        self.model_0th = copy.deepcopy(model)
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg

        self.aggregator_func = self.__dos_aggregation
        self.cpd_l2 = COPOD()
        self.cpd_cs = COPOD()
        print("Defense: Copod-DOS")


        if len(self.byztn_cfg) != 0:
            byz_clients = [int(b) for b in self.byztn_cfg["byztn_clients"]]
            if id in byz_clients:
                self.byz_way = globals()[self.byztn_cfg["byztn_method"]](cfg, model)
                print("Byz Method", self.byztn_cfg["byztn_method"])



    #-------- Server methods ----------

    def __dos_aggregation(self, lsets):
        wvecs = [fedops.get_param_from_state(l["model"].state_dict())
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

        final_wvec = torch.sum(cweighed_wvec, axis = 0)

        agg_states = fedops.set_param_in_state(lsets[0]["model"].state_dict(), final_wvec)

        info_dict = {"client_weights":cweigh.flatten().tolist()}
        return agg_states, info_dict


##------------------------------------------------------------------------------