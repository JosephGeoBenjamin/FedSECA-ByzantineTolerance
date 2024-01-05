# Ξ - chi used as underscore in Classes

import copy
import torch
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

        model.load_state_dict(gset["model"].state_dict(), strict=True)
        model = model.to(device)
        ghatch = {}

        return model, ghatch


    #-------- Server methods ----------

    def __plain_fedavg(self, lsets):
        local_states = []
        for ls in lsets:
            local_states.append(ls["model"].state_dict())
        agg_states = fedops.global_average_statedict(local_states)

        return agg_states


    # @instancemethod  #Global calculation to send to locals
    def aggregate_globally(self, lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        if not device: device = next(self.model_0th.parameters()).device
        for ls in lsets: ls["model"].to(device) #for nn.module .cuda is both inplace and assignable

        agg_model = fedops.model_copier(lsets[0]["model"])

        agg_states = self.aggregator_func(lsets)

        agg_model.load_state_dict(agg_states)

        gset = {"model": agg_model}
        return gset


##------------------------------------------------------------------------------


class KrumΞByzantine(NoGuardΞByzantine):

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
        return agg_states


##------------------------------------------------------------------------------