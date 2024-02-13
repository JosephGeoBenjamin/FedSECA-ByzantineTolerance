import copy

import torch
import torch.nn.functional as torch_F
import numpy as np
import scipy.stats as spstats
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
    Reference: Suppressing Poisoning Attacks on Federated Learning for Medical Imaging.
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
        attack_wvec =  wvec_current - (self.lmbd*S)

        out_state = fedops.set_param_in_state(lmodel_state_tth, attack_wvec,
                                    keys_to_ignore=self.vec_state_ignore)
        self.wvec_tminus1 = wvec_current.clone()

        return out_state


class OzfaturaROPζAttack():
    """
    Reference: Byzantines can also Learn from History: Fall of Centered Clipping in Federated Learning
    modified on basecode from authors
    """

    def __init__(self, cfg, model_at_start):
        self.byz_cfg = cfg.byztn_cfg
        self.z_max = self.byz_cfg.get("z_max")
        self.pi    = self.byz_cfg.get("pi_angle") # angle
        self.rho   = self.byz_cfg.get("rho_reloc") # relocation
        self.lmbd  = self.byz_cfg.get("lambda_refwt") # reference weightage

        self.num_client_k = n = cfg["data_centers_count"]
        self.num_byzant_b = m = len(self.byz_cfg["byztn_clients"])

        self.vec_state_ignore = ["num_batches_tracked"]
        self.first_step_ignore = True

        ## this is global common start point
        self.gwvec_0th:torch.Tensor = fedops.get_param_from_state(model_at_start.state_dict(),
                                    keys_to_ignore=self.vec_state_ignore)
        self.gwvec_tminus1:torch.Tensor = copy.deepcopy(self.gwvec_0th)

        if not self.z_max:
            s = np.floor(n / 2 + 1) - m
            cdf_value = (n - m - s) / (n - m)
            self.z_max = spstats.norm.ppf(cdf_value)


    def modify(self, lmodel_state_tth, gmodel_state_tminus1):

        ## Benign Gradients:-> torch.mean(benign_gradients, 1)
        # original work uses "mean of the benign gradients", exactness only possible in omniscient case
        # here we use local model after an epoch following local protocol
        m_t = fedops.get_param_from_state(copy.deepcopy(lmodel_state_tth),
                                    keys_to_ignore=self.vec_state_ignore)

        ## Global reference:-> ud = self.global_momentum.clone()
        # attacker has only access to global model recieved from Server, assuming they are not omniscient
        m_tminus1 = fedops.get_param_from_state(gmodel_state_tminus1,
                                    keys_to_ignore=self.vec_state_ignore)


        ## reference point, global momentum
        # here we can take 0th or (t-1)th based on clipping defense used
        ud = m_tminus1.clone()
        # if first iteration, set the reference point to the mean of the benign momentums
        if self.first_step_ignore:
            ud = m_t.clone()
            self.first_step_ignore = False

        ## Target point of attack ; between previous and current
        ud = (ud * self.lmbd) + (m_t * (1-self.lmbd))

        ## Orthogonal vector
        pert = torch.ones_like(m_t)  # inital perturbation
        proj_pert = ud * ((pert @ ud) / (ud @ ud)) # vector projection
        pert = pert - proj_pert # vector orthogonal to ud (rejecting the projection)

        ## setting perturbation Angle
        angle = self.pi # desired angle of the perturbation
        sin, cos = np.sin(angle*np.pi/180), np.cos(angle* np.pi/180)
        n_ud = ud / ud.norm() # normalised reference point
        n_pert = pert / pert.norm() # normalised pert
        pert = (n_pert * sin + n_ud * cos) # rotate the perturbation w.r.t global momentum

        ## setting perturbation Scale
        pert = pert * (self.z_max / pert.norm()) # scale the perturbation

        ## The Attack
        attack_location = (m_tminus1 * self.rho) + (m_t * (1-self.rho)) #relocate reference
        attack_wvec = attack_location + pert # final attack added to desired location

        out_state = fedops.set_param_in_state(copy.deepcopy(lmodel_state_tth), attack_wvec,
                                    keys_to_ignore=self.vec_state_ignore)

        return out_state
