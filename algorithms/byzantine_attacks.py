import copy

import torch
import torch.nn.functional as torch_F
import numpy as np

import algorithms.federation_ops as fedops



##==============================================================================
## ATTACKS
"""
model_init/model_at_start -> model recieved init communication i.e at very first broadcast of params for training
gmodel_state_tminus1 -> model recieved at Tth aggregation round from Server
lmodel_state_tth -> gmodel_state_tminus1 updated for one set LocalRounds before sending to Server

Tth: Ginit->L0->G0->L1->G1...->LT->GT->L
"""

class RandomizedζAttack():
    def __init__(self, cfg, model_at_start ):
        # self.model_init = copy.deepcopy(model)
        self.byz_cfg = cfg.byztn_cfg

    def modify(self, lmodel_state_tth, gmodel_state_tminus1):
        weight_vec = fedops.get_param_from_state(lmodel_state_tth)
        weight_vec[:] = torch.rand(len(weight_vec))
        out_state = fedops.set_param_in_state(lmodel_state_tth, weight_vec)

        return out_state


class AffineζAttack():
    def __init__(self, cfg, model_at_start):
        self.byz_cfg = cfg.byztn_cfg
        self.scaler = cfg.byztn_cfg["scale"]

    def modify(self, lmodel_state_tth, gmodel_state_tminus1):
        weight_vec = fedops.get_param_from_state(lmodel_state_tth)
        weight_vec[:]= self.scaler * weight_vec
        out_state = fedops.set_param_in_state(lmodel_state_tth, weight_vec)

        return out_state


class NaifttCraftedζAttack():
    """
    Reference: "Suppressing Poisoning Attacks on Federated Learning for Medical Imaging."
    Taken from: https://github.com/Naiftt/SPAFD/
    """

    def __init__(self, cfg, model_at_start):
        self.byz_cfg = cfg.byztn_cfg
        self.lmbd = cfg.byztn_cfg["lambda"] # 0.1 in paper

        self.vec_state_ignore = ["num_batches_tracked"]

        self.wvec_tminus1 = fedops.get_param_from_state(model_at_start.state_dict(),
                                            keys_to_ignore=self.vec_state_ignore)


    def modify(self, lmodel_state_tth, gmodel_state_tminus1):
        wvec_current = fedops.get_param_from_state(lmodel_state_tth,
                                keys_to_ignore=self.vec_state_ignore)

        S = (wvec_current > self.wvec_tminus1).long()
        S[S==0] = -1
        new_wvec =  wvec_current - (self.lmbd*S)

        out_state = fedops.set_param_in_state(lmodel_state_tth, new_wvec,
                                    keys_to_ignore=self.vec_state_ignore)
        self.wvec_tminus1 = wvec_current.clone()

        return out_state


