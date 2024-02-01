# Ξ - chi used as underscore in Classes

import copy
import torch

from sketching.count_sketch import CountSketchVec, CountSketchVec_NoSignHash


##------------------------------------------------------------------------------
class MethodsΞTemplate():
    def __init__(self, cfg, id, model, device="cpu"):
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

    # @instancemethod  #process global info for local use
    def desynopsize_local(self, gset, device=None, model_struct=None): #used at end of local round at each client
        """ Return: lstat
        """
        dgset = {}
        return dgset

    @staticmethod #
    def compute_local_deviation(zxs, agghatch, cen_id=None): #at each client
        """
        """
        loss, print_info = torch.tensor(0), {}
        return loss, print_info


    # @instancemethod  #Global calculation to send to locals
    def aggregate_globally(self, lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        gset = {}
        return gset



##==============================================================================
## COMMONS

def model_copier(m):
    """a safer wrapper, to be enabled or disabled based on debug"""
    return copy.deepcopy(m)


def global_average_statedict(w:list, device = "cpu"):
    """
    w: list of pytorch weights statedict
    Returns the average of the weights as statedict
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key].to(device)
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + w[i][key].to(device)
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def get_param_from_model(model:torch.nn.Module, only_with_grad=False):
    param_vec = []
    for p in model.parameters():
        if ( not only_with_grad) or p.requires_grad:
            param_vec.append(p.data.view(-1).float())
    return torch.cat(param_vec)


def get_param_from_state(state_dict:dict, keys_to_ignore:list=[]):
    """ keys_to_ignore:  can be list subset string or full key name
    """
    param_vec = []
    for key, value in state_dict.items():
        if ( sum([i in key for i in keys_to_ignore]) == 0 ):
            param_vec.append(value.view(-1).float())
    return torch.cat(param_vec)


def set_param_in_model(model, param_vec, only_with_grad=False):
    """ Does inplace change to model object and also returns
    """
    start = 0
    for p in model.parameters():
        if ( not only_with_grad) or p.requires_grad:
            end = start + p.numel()
            p.data.zero_()
            p.data.add_(param_vec[start:end].view(p.size()))
            start = end
    assert (end == len(param_vec)), f"Mismatch in Sizes in set_param : {end} vs {len(param_vec)}"
    return model


def set_param_in_state(state_dict, param_vec, keys_to_ignore:list=[]):
    """ No inplace; only rely on return
    keys_to_ignore:  can be list subset string or full key name
    """
    start = 0
    for key, value in state_dict.items():
        if ( sum([i in key for i in keys_to_ignore]) == 0 ):
            # param_vec.append(value.view(-1).float())
            end = start + value.view(-1).shape[0]
            state_dict[key] = torch.clone(param_vec[start:end].view(value.shape))
            start = end
    assert (end == len(param_vec)), f"Mismatch in Sizes in set_param : {end} vs {len(param_vec)}"
    return state_dict



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


##==============================================================================


class SimpleStateΞFedAvg():

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.model_0th = copy.deepcopy(model)

    #-------- Client methods ----------

    # @instancemethod #Locals calculation to send to Global
    def synopsize_local(self, zxs): #used at end of local round at each client
        """ zxs: {"model", }
        """
        lset = {}
        lset["model"] = model_copier(zxs["model"])

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

    # @instancemethod  #Global calculation to send to locals
    def aggregate_globally(self, lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        if not device: device = next(self.model_0th.parameters()).device
        for ls in lsets: ls["model"].to(device) #for nn.module .cuda is both inplace and assignable

        agg_model = model_copier(lsets[0]["model"])
        local_states = []
        for ls in lsets:
            local_states.append(ls["model"].state_dict())
        agg_states = global_average_statedict(local_states)

        agg_model.load_state_dict(agg_states)

        gset = {"model": agg_model}
        return gset


##------------------------------------------------------------------------------

class SimpleParamΞFedAvg():

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.model_0th = copy.deepcopy(model)

    #-------- Client methods ----------

    # @instancemethod #Locals calculation to send to Global
    def synopsize_local(self, zxs): #used at end of local round at each client
        """ zxs: {"model", }
        """
        lset = {}
        lset["param_vec"] = get_param_from_state(zxs["model"].state_dict())

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
        model = copy.deepcopy(model_struct).to(device)
        if not gset: return model, {}

        new_state = set_param_in_state(model.state_dict(), gset["param_vec"])
        model.load_state_dict(new_state, strict=True)
        ghatch = {}

        return model, ghatch


    #-------- Server methods ----------

    # @instancemethod  #Global calculation to send to locals
    def aggregate_globally(self, lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        if not device: device = next(self.model_0th.parameters()).device

        for ls in lsets: ls["param_vec"].to(device)

        agg_vec = lsets[0]["param_vec"].clone()
        for ls in lsets[1:]:
            agg_vec += ls["param_vec"]
        agg_vec /= len(lsets)

        gset = {"param_vec": agg_vec}
        return gset

##------------------------------------------------------------------------------

class DeltaParamΞFedAvg():

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.model_tminus_1 = copy.deepcopy(model)

    #-------- Client methods ----------

    # @instancemethod #Locals calculation to send to Global
    def synopsize_local(self, zxs): #used at end of local round at each client
        """ zxs: {"model", }
        """
        lset = {}
        newvec = get_param_from_state(zxs["model"].state_dict())
        oldvec = get_param_from_state(self.model_tminus_1.state_dict())
        lset["delta_vec"] = newvec - oldvec

        return lset

    #-------- Shared methods ----------

    # @instancemethod #process global info for local use
    def desynopsize_local(self, gset, device=None, model_struct=None): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"model", }
        """
        model_struct = self.model_tminus_1 if not model_struct else model_struct
        if not device: device = next(model_struct.parameters()).device

        model = copy.deepcopy(model_struct) # to prevent unintend model changes
        if not gset: return model, {}

        oldvec = get_param_from_state(self.model_tminus_1.state_dict())
        newvec = oldvec + gset["delta_vec"].to(device)

        new_state = set_param_in_state(model.state_dict(), newvec)
        model.load_state_dict(new_state, strict=True)

        self.model_tminus_1 = copy.deepcopy(model)
        ghatch = {}
        return model, ghatch


    #-------- Server methods ----------

    # @instancemethod  #Global calculation to send to locals
    def aggregate_globally(self, lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        if not device: device = next(self.model_tminus_1.parameters()).device

        for ls in lsets: ls["delta_vec"].to(device)

        agg_vec = lsets[0]["delta_vec"].clone()
        for ls in lsets[1:]:
            agg_vec += ls["delta_vec"]
        agg_vec /= len(lsets)

        gset = {"delta_vec": agg_vec}
        return gset


##==============================================================================

## NOTE: only statedict will have running BN stat, when accessed as params it wil not showup
## For sketching using running stat will corrupt that bin to whihc it got mapped

SketchMethodVarient = CountSketchVec

class NaiveΞCountSketch():

    #-------- Stateful variables Local ------
    def __init__(self, cfg, id, model, device="cpu"):
        self.device = device
        self.id = id
        self.model_0th = copy.deepcopy(model)

        vec_size = len(get_param_from_model(model))
        cols = vec_size // cfg.sketch_compress_factor
        rows = cfg.sketch_hashes
        print(vec_size, cols, rows)

        self.csobj = SketchMethodVarient(d=vec_size, c=cols, r=rows, device=device)

        print("Naive Sketch implementation")

    #-------- Client methods ----------

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
    # @instancemethod #process global info for local use
    def desynopsize_local(self, gset, device=None, model_struct=None): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"sketch", }
        """
        model_struct = self.model_0th if not model_struct else model_struct
        if not device: device = next(model_struct.parameters()).device

        ## since no compression or sketching used
        model = copy.deepcopy(model_struct)
        if not gset: return model, {}

        us_params = gset["sketch"].unSketch(all=True)

        model = set_param_in_model(model, us_params.to(device))

        ghatch = {}
        return model, ghatch

    #-------- Server methods ----------

    # @instancemethod  #Global calculation to send to locals
    def aggregate_globally(self, lsets, device=None): #used at begining of local round central
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

##------------------------------------------------------------------------------

class DeltaWeightΞCountSketch(NaiveΞCountSketch):
    """ DeltaWeightΞCountSketch
        Local and Global are expected to be Decoupled from each other
        since no direct averaing update is observed
    """

    #-------- Stateful variables Local ------
    def __init__(self, cfg, id, model, device="cpu"):
        super().__init__(cfg, model, device)
        self.device = device
        self.id = id
        self.model_0th      = copy.deepcopy(model).to(device)
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

        return lset


    #-------- Shared methods ----------
    # @instancemethod  #process global info for local use
    def desynopsize_local(self, gset, device=None, model_struct=None): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"sketch", }
        """
        model_struct = self.model_0th if not model_struct else model_struct
        if not device: device = next(model_struct.parameters()).device

        ## since no compression or sketching used
        model = copy.deepcopy(model_struct)
        if not gset: return model, {}

        delta_us_params = gset["sketch"].unSketch(all=True)
        param_vec   = get_param_from_model(self.model_tminus_1)

        updated_param_vec = param_vec + delta_us_params.to(device)

        model = set_param_in_model(model, updated_param_vec)

        self.model_tminus_1 = copy.deepcopy(model)
        ghatch = {}
        return model, ghatch

##------------------------------------------------------------------------------

class DeltaWeightBNΞCountSketch(NaiveΞCountSketch):
    """ DeltaWeightBNΞCountSketch
        Local and Global are expected to be Decoupled from each other
        since no direct averaing update is observed
    """

    #-------- Stateful variables Local ------
    def __init__(self, cfg, id, model, device="cpu"):
        super().__init__(cfg, id, model, device)
        self.device = device
        self.id = id
        self.model_0th      = copy.deepcopy(model).to(device)
        self.model_tminus_1 = copy.deepcopy(model).to(device)


        print("Delta Weight Sketch implementation")

    def get_bn_params(self, model):
        m1state = model.state_dict()
        bn_states = {}
        for k in m1state:
            if ".bn" in k:
                bn_states[k] = m1state[k]
        return bn_states


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
        lset["BN"] = self.get_bn_params(zxs["model"])
        self.csobj.zero()

        return lset


    #-------- Shared methods ----------
    # @instancemethod  #process global info for local use
    def desynopsize_local(self, gset, device=None, model_struct=None): #used at end of local round at each client
        """ model_struct: torch nn.module object
            gset: global aggregations {"sketch", }
        """
        model_struct = self.model_0th if not model_struct else model_struct
        if not device: device = next(model_struct.parameters()).device

        ## since no compression or sketching used
        model = copy.deepcopy(model_struct)
        if not gset: return model, {}

        delta_us_params = gset["sketch"].unSketch(all=True)
        param_vec   = get_param_from_model(self.model_tminus_1)

        updated_param_vec = param_vec + delta_us_params.to(device)

        model = set_param_in_model(model, updated_param_vec)

        new_state = model.state_dict()
        new_state.update(gset["BN"])
        model.load_state_dict(new_state)

        self.model_tminus_1 = copy.deepcopy(model)
        ghatch = {}
        return model, ghatch


    # @instancemethod  #Global calculation to send to locals
    def aggregate_globally(self, lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        if not device: device = lsets[-1]["sketch"].device
        for ls in lsets: ls["sketch"].to_(device)

        agg_sketch = model_copier(lsets[0]["sketch"])
        for ls in lsets[1:]:
            agg_sketch += ls["sketch"]
        agg_sketch = agg_sketch / len(lsets)

        gset = {"sketch": agg_sketch}
        gset["BN"] = global_average_statedict([l['BN'] for l in lsets], device=device)
        # del lsets
        return gset

##------------------------------------------------------------------------------

class FetchSGDishΞCountSketch(DeltaWeightΞCountSketch):
    """ Thin implementation on FetchSGD: https://arxiv.org/abs/2007.07682 """

    #--------- Stateful variables Local -------
    def __init__(self, cfg, id, model, device="cpu"):
        super().__init__(cfg, id, model, device)
        self.device = device
        self.id = id
        self.rho = 0.9
        self.eta = 0.9
        self.topk_ratio = 0.1
        self.err_sketch = None
        self.mom_sketch = None

        print("Fetch SGD implementation")

    #--------- Server methods -----------

    # @instancemethod  #Global calculation to send to locals
    def aggregate_globally(self, lsets, device=None): #used at begining of local round central
        """ Return: aggregated stat
        """
        if not device: device = lsets[-1]["sketch"].device
        for ls in lsets: ls["sketch"].to_(device)

        ## get adj_sketch, self.err_sketch, self.mom_sketch
        adj_sketch = self._setup_cls_sketches(lsets[-1]["sketch"])

        ## sketches summation
        agg_sketch = model_copier(lsets[0]["sketch"])
        for ls in lsets[1:]:
            agg_sketch += ls["sketch"]
        agg_sketch = agg_sketch / len(lsets)

        ## momentum term
        self.mom_sketch =  self.mom_sketch * self.rho + agg_sketch

        ## topK unsketched delta vector
        topk_count = int(self.topk_ratio * agg_sketch.d)
        tkuSv = (agg_sketch * self.eta + self.err_sketch).unSketch(k=topk_count)  ## if topk remove be mindful to subtract instead of zeroing
        adj_sketch.accumulateVec(tkuSv)

        ## error term
        # self.err_sketch = self.eta*agg_sketch + self.err_sketch - adj_sketch     ## -- In theory subtract

        self.err_sketch.table = torch.where(adj_sketch.table !=0,
                                           0, self.err_sketch.table)   ## -- practise  elements set to zero
        self.err_sketch = self.mom_sketch * self.eta + self.err_sketch

        gset = {"sketch": adj_sketch}
        # del lsets
        return gset

    @classmethod
    def _setup_cls_sketches(self, proto_sketch):

        if not torch.is_tensor(self.err_sketch ):
            self.err_sketch = copy.deepcopy(proto_sketch)
            self.err_sketch.zero()
        if not torch.is_tensor(self.mom_sketch ):
            self.mom_sketch = copy.deepcopy(proto_sketch)
            self.mom_sketch.zero()
        adj_sketch = copy.deepcopy(proto_sketch)
        adj_sketch.zero()
        return adj_sketch


##******************************************************************************
