import torch
import torch.nn.functional as torch_F
import numpy as np

import algorithms.federation_ops as fedops



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
    """
    Reference: "Suppressing Poisoning Attacks on Federated Learning for Medical Imaging."
    Taken from: https://github.com/Naiftt/SPAFD/
    """

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


