# Ξ - chi used as underscore in Classes

import copy
import torch
from torch import nn


from sketching.count_sketch import CountSketchVec

##------------------------------------------------------------------------------
class MethodsΞTemplate():
    def __init__(self, cfg, model, device="cpu"):
        pass

    #@instancemethod # Local info aggregation Locally
    def accumulate_locals(self, zxs={}): # tobe used with in a round/epoch at each client
        return {}

    #@instancemethod #Locals calculation to send to Global
    def synopsize_local(self, zxs={}): #used at end of local round at each client
        """ Return: lstat
        """
        lset = {}
        return lset

    @staticmethod #process global info for local use
    def desynopsize_local(gset): #used at end of local round at each client
        """ Return: lstat
        """
        dgset = {}
        return dgset

    @staticmethod #
    def compute_local_deviation(lset, gset, cen_id=None): #at each client
        """
        """
        loss, print_info = torch.tensor(0), {}
        return loss, print_info


    @staticmethod  #Global calculation to send to locals
    def aggregate_globally(lsets): #used at begining of local round central
        """ Return: aggregated stat
        """
        gset = {}
        return gset



##==============================================================================
## COMMONS

def model_copier(m):
    return copy.deepcopy(m)


def global_average_weights(w:list, device = "cpu"):
    """
    w: list of pytorch parameters for weighs
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key].to(device)
        for i in range(1, len(w)):
            w_avg[key] += w[i][key].to(device)
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def get_param_from_model(model:torch.nn.Module):
    param_vec = []
    for p in model.parameters():
        if p.requires_grad:
            param_vec.append(p.data.view(-1).float())
    return torch.cat(param_vec)

def get_param_from_state(state_dict:dict):
    param_vec = []
    for key, value in state_dict.items():
        param_vec.append(value.view(-1).float())
    return torch.cat(param_vec)


def get_topK_param(model, K):
    # top K both positive and negative (k/2 each)
    vec = get_param_from_model(model)
    out_vec = torch.zeros(vec.shape).to(vec.device)
    v1, i1 = torch.topk(vec, k= K//2+K%2)
    v2, i2 = torch.topk(vec, k= K//2, largest=False)
    out_vec[i1] = v1
    out_vec[i2] = v2
    return out_vec


def set_param_in_model(model, param_vec):
    start = 0
    for p in model.parameters():
        if p.requires_grad:
            end = start + p.numel()
            p.data.zero_()
            p.data.add_(param_vec[start:end].view(p.size()))
            start = end
    assert (end == len(param_vec)), f"Mismatch in Sizes in set_param : {end} vs {len(param_vec)}"
    return model

##==============================================================================


class SimpleΞFedAvg():

    def __init__(self, cfg, model, device="cpu"):
        pass

    #-------- Client methods ----------
    # @instancemethod  # Local info aggregation Locally
    def accumulate_locals(self, zxs): # tobe used with in a round/epoch at each client
        pass

    # @instancemethod #Locals calculation to send to Global
    def synopsize_local(self, zxs): #used at end of local round at each client
        """ zxs: {"model", }
        """
        lset = {}
        lset["model"] = model_copier(zxs["model"])
        return lset

    @staticmethod #process global info for local use
    def desynopsize_local(model_struct, gset): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"model", }
        """
        ## since no compression or sketching used
        model = model_copier(model_struct)
        if not gset: return model, {}

        ghatch = {}

        return model, ghatch

    @staticmethod #
    def compute_local_deviation(zxs, agghatch, cen_id=None): #at each client
        """
        """
        loss, print_info = torch.tensor(0), {}
        return loss, print_info

    #-------- Server methods ----------

    @staticmethod  #Global calculation to send to locals
    def aggregate_globally(lsets): #used at begining of local round central
        """ Return: aggregated stat
        """
        local_models = []
        for ls in lsets:
            local_models.append(model_copier(ls["model"]))
        agg_model = global_average_weights(local_models)

        gset = {"model": agg_model}
        return gset


class NaiveΞCountSketch():

    #-------- Stateful variables Local ------
    def __init__(self, cfg, model, device="cpu"):

        vec_size = len(get_param_from_model(model))
        cols = vec_size // cfg.sketch_compress_factor
        rows = cfg.sketch_hashes
        print(vec_size, cols, rows)

        self.csobj = CountSketchVec(d=vec_size, c=cols, r=rows, device=device)


    #-------- Client methods ----------

    # @instancemethod  # Local info aggregation Locally
    def accumulate_locals(self, zxs): # tobe used with in a round/epoch at each client
        pass

    # @instancemethod #Locals calculation to send to Global
    def synopsize_local(self, zxs): #used at end of local round at each client
        """ zxs: {"model", }
        """
        lset = {}
        param_vec = get_param_from_model(zxs["model"])
        self.csobj.accumulateVec(param_vec)

        lset["sketch"] = model_copier(self.csobj)
        self.csobj.zero()
        return lset

    @staticmethod #process global info for local use
    def desynopsize_local(model_struct, gset): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"sketch", }
        """
        ## since no compression or sketching used
        model = model_copier(model_struct)
        if not gset: return model, {}

        us_params = gset["sketch"].unSketch(all=True)

        model = set_param_in_model(model, us_params)

        ghatch = {}
        return model, ghatch

    @staticmethod #
    def compute_local_deviation(zxs, agghatch, cen_id=None): #at each client
        """
        """
        loss, print_info = torch.tensor(0), {}
        return loss, print_info

    #-------- Server methods ----------

    @staticmethod  #Global calculation to send to locals
    def aggregate_globally(lsets): #used at begining of local round central
        """ Return: aggregated stat
        """
        agg_sketch = lsets[0]["sketch"]
        for ls in lsets[1:]:
            agg_sketch += ls["sketch"]
        agg_sketch = agg_sketch / len(lsets)
        gset = {"sketch": agg_sketch}
        return gset