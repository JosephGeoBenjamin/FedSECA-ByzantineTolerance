# Ξ - chi used as underscore in Classes

import copy
import torch

from sketching.count_sketch import CountSketchVec, CountSketchVec_NoSignHash


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

    # Strict Static
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
    """a safer wrapper, to be enabled or disabled based on debug"""
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
    vec = get_param_from_model(model)
    out_vec = torch.zeros(vec.shape).to(vec.device)

    ### # top K both positive and negative (k/2 each)
    # v1, i1 = torch.topk(vec, k= K//2+K%2, sorted=False)
    # v2, i2 = torch.topk(vec, k= K//2, largest=False, sorted=False)
    # out_vec[i1] = v1
    # out_vec[i2] = v2

    tv1, ti1 = torch.topk(vec**2, k=K, sorted=False)
    out_vec[ti1] = vec[ti1]

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

    @staticmethod #
    def compute_local_deviation(zxs, agghatch, cen_id=None): #at each client
        """
        """
        loss, print_info = torch.tensor(0), {}
        return loss, print_info

    #-------- Shared methods ----------

    @staticmethod #process global info for local use
    def desynopsize_local(model_struct, gset, device=None): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"model", }
        """
        if not device: device = next(model_struct.parameters()).device

        ## since no compression or sketching used
        model = copy.deepcopy(model_struct)
        if not gset: return model, {}

        model = model_copier(gset["model"]).to(device)
        ghatch = {}

        return model, ghatch

    #-------- Server methods ----------

    @staticmethod  #Global calculation to send to locals
    def aggregate_globally(lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        if not device: device = next(lsets[-1]["model"].parameters()).device
        for ls in lsets: ls["model"].to(device) #for nn.module .cuda is both inplace and assignable

        agg_model = model_copier(lsets[0]["model"])
        local_states = []
        for ls in lsets:
            local_states.append(ls["model"].state_dict())
        agg_states = global_average_weights(local_states)

        agg_model.load_state_dict(agg_states)

        gset = {"model": agg_model}
        return gset

##==============================================================================

SketchMethodVarient = CountSketchVec_NoSignHash

class NaiveΞCountSketch():

    #-------- Stateful variables Local ------
    def __init__(self, cfg, model, device="cpu"):
        self.device = device

        vec_size = len(get_param_from_model(model))
        cols = vec_size // cfg.sketch_compress_factor
        rows = cfg.sketch_hashes
        print(vec_size, cols, rows)

        self.csobj = SketchMethodVarient(d=vec_size, c=cols, r=rows, device=device)

        print("Naive Sketch implementation")

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

        lset["sketch"] = copy.deepcopy(self.csobj)
        self.csobj.zero()
        return lset

    #-------- Shared methods ----------
    # Strict Static
    @staticmethod #process global info for local use
    def desynopsize_local(model_struct, gset, device=None): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"sketch", }
        """
        if not device: device = next(model_struct.parameters()).device

        ## since no compression or sketching used
        model = copy.deepcopy(model_struct)
        if not gset: return model, {}

        us_params = gset["sketch"].unSketch(all=True)

        model = set_param_in_model(model, us_params.to(device))

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
    def aggregate_globally(lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        if not device: device = lsets[-1]["sketch"].device
        for ls in lsets: ls["sketch"].to_(device)

        agg_sketch = model_copier(lsets[0]["sketch"])
        for ls in lsets[1:]:
            agg_sketch += ls["sketch"]
        agg_sketch = agg_sketch / len(lsets)

        gset = {"sketch": agg_sketch}
        # del lsets
        return gset


class DeltaWeightΞCountSketch(NaiveΞCountSketch):
    """ DeltaWeightΞCountSketch
        Local and Global are expected to be Decoupled from each other
        since no direct averaing update is observed
    """

    #-------- Stateful variables Local ------
    def __init__(self, cfg, model, device="cpu"):
        super().__init__(cfg, model, device)
        self.device = device
        self.model_tminus_1 = copy.deepcopy(model).to(device)

        print("Delta Weight Sketch implementation")

    #-------- Client methods ----------

    # @instancemethod #Locals calculation to send to Global
    def synopsize_local(self, zxs): #used at end of local round at each client
        """ zxs: {"model", }
        """
        lset = {}
        old_param_vec = get_param_from_model(self.model_tminus_1)
        new_param_vec = get_param_from_model(zxs["model"])

        delta_param_vec = new_param_vec - old_param_vec
        self.csobj.accumulateVec(delta_param_vec)

        lset["sketch"] = copy.deepcopy(self.csobj)

        self.csobj.zero()
        self.model_tminus_1 = copy.deepcopy(zxs["model"])

        return lset


    #-------- Shared methods ----------
    # Strict Static
    @staticmethod #process global info for local use
    def desynopsize_local(model_struct, gset, device=None): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"sketch", }
        """
        if not device: device = next(model_struct.parameters()).device

        ## since no compression or sketching used
        model = copy.deepcopy(model_struct)
        if not gset: return model, {}

        delta_us_params = gset["sketch"].unSketch(all=True)
        param_vec   = get_param_from_model(model_struct)

        updated_param_vec = param_vec + delta_us_params.to(device)

        model = set_param_in_model(model, updated_param_vec)

        ghatch = {}
        return model, ghatch


class FetchSGDishΞCountSketch(DeltaWeightΞCountSketch):
    """ Thin implementation on FetchSGD: https://arxiv.org/abs/2007.07682 """

    #--------- Stateful variables Local -------
    def __init__(self, cfg, model, device="cpu"):
        super().__init__(cfg, model, device)

        print("Fetch SGD implementation")

    #--------- Server methods -----------

    rho = 0.9
    eta = 0.9
    topk_ratio = 0.1
    err_sketch = None
    mom_sketch = None


    @classmethod  #Global calculation to send to locals
    def aggregate_globally(cls, lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        if not device: device = lsets[-1]["sketch"].device
        for ls in lsets: ls["sketch"].to_(device)

        ## get adj_sketch, cls.err_sketch, cls.mom_sketch
        adj_sketch = cls._setup_cls_sketches(lsets[-1]["sketch"])

        ## sketches summation
        agg_sketch = model_copier(lsets[0]["sketch"])
        for ls in lsets[1:]:
            agg_sketch += ls["sketch"]
        agg_sketch = agg_sketch / len(lsets)

        ## momentum term
        cls.mom_sketch =  cls.mom_sketch * cls.rho + agg_sketch

        ## topK unsketched delta vector
        topk_count = int(cls.topk_ratio * agg_sketch.d)
        tkuSv = (agg_sketch * cls.eta + cls.err_sketch).unSketch(k=topk_count)  ## if topk remove be mindful to subtract instead of zeroing
        adj_sketch.accumulateVec(tkuSv)

        ## error term
        # cls.err_sketch = cls.eta*agg_sketch + cls.err_sketch - adj_sketch     ## -- In theory subtract

        cls.err_sketch.table = torch.where(adj_sketch.table !=0,
                                           0, cls.err_sketch.table)   ## -- practise  elements set to zero
        cls.err_sketch = cls.mom_sketch * cls.eta + cls.err_sketch

        gset = {"sketch": adj_sketch}
        # del lsets
        return gset

    @classmethod
    def _setup_cls_sketches(cls, proto_sketch):
        if not torch.is_tensor(cls.err_sketch ):
            cls.err_sketch = copy.deepcopy(proto_sketch)
            cls.err_sketch.zero()
        if not torch.is_tensor(cls.mom_sketch ):
            cls.mom_sketch = copy.deepcopy(proto_sketch)
            cls.mom_sketch.zero()
        adj_sketch = copy.deepcopy(proto_sketch)
        adj_sketch.zero()
        return adj_sketch


##******************************************************************************
