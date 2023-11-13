# Ξ - chi used as underscore in Classes

import copy
import torch
from torch import nn

##------------------------------------------------------------------------------
class MethodsΞTemplate():
    @staticmethod # Local info aggregation Locally
    def accumulate_locals(zxs={}): # tobe used with in a round/epoch at each client
        return {}

    @staticmethod #Locals calculation to send to Global
    def synopsize_local(zxs={}): #used at end of local round at each client
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


def get_param_vec(model):
    param_vec = []
    for p in model.parameters():
        if p.requires_grad:
            param_vec.append(p.data.view(-1).float())
    return torch.cat(param_vec)


def get_topK_param_vec(model, K):
    # top K both positive and negative (k/2 each)
    vec = get_param_vec(model)
    out_vec = torch.zeros(vec.shape).to(vec.device)
    v1, i1 = torch.topk(vec, k= K//2+K%2)
    v2, i2 = torch.topk(vec, k= K//2, largest=False)
    out_vec[i1] = v1
    out_vec[i2] = v2
    return out_vec

##==============================================================================


class SimpleΞFedAvg():

    #-------- Client methods ----------

    @staticmethod # Local info aggregation Locally
    def accumulate_locals(zxs={}): # tobe used with in a round/epoch at each client
        return {}

    @staticmethod #Locals calculation to send to Global
    def synopsize_local(zxs={}): #used at end of local round at each client
        """ zxs: {"model", }
        """
        lset = {}
        lset["weight_state"] = copy.deepcopy(zxs["weight_state"])
        return lset

    @staticmethod #process global info for local use
    def desynopsize_local(model_struct, gset): #used at end of local round at each client
        """ Return: lstat
        """
        ## since no compression or sketching used
        model = copy.deepcopy(model_struct)
        if not gset: return model, {}

        model.load_state_dict(gset["weight_state"])
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
        local_weights = []
        for ls in lsets:
            local_weights.append(ls["weight_state"])
        global_weight_state = global_average_weights(local_weights)

        gset = {"weight_state": global_weight_state}
        return gset


    #-------- Internal functions ----------

