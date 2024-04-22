import copy

import torch
import torch.nn.functional as torch_F
import numpy as np
import random
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


class LabelFlipζAttack():
    def __init__(self, cfg, model_at_start, device="cpu" ):
        # self.model_init = copy.deepcopy(model)
        print("This is a dummy init; LabelFlipζAttack for Training Phase attack")

    def modify(self, lmodel_state_tth, gmodel_state_tminus1, omniscience={}):
        return lmodel_state_tth


class RandomizedζAttack():
    def __init__(self, cfg, model_at_start, device="cpu" ):
        # self.model_init = copy.deepcopy(model)
        self.device = device
        self.byz_cfg = cfg.byztn_cfg

        print("ATTACK: RandomizedζAttack")

    def modify(self, lmodel_state_tth, gmodel_state_tminus1, omniscience={}):
        weight_vec = fedops.get_param_from_state(lmodel_state_tth)
        weight_vec[:] = torch.rand(len(weight_vec))
        out_state = fedops.set_param_in_state(lmodel_state_tth, weight_vec)

        return out_state


class AffineζAttack():
    def __init__(self, cfg, model_at_start,  device="cpu"):
        self.device = device
        self.byz_cfg = cfg.byztn_cfg
        self.scaler = cfg.byztn_cfg["scale"]

        print("ATTACK: AffineζAttack")

    def modify(self, lmodel_state_tth, gmodel_state_tminus1, omniscience={}):
        weight_vec = fedops.get_param_from_state(lmodel_state_tth)
        weight_vec[:]= self.scaler * weight_vec
        out_state = fedops.set_param_in_state(lmodel_state_tth, weight_vec)

        return out_state


class FangCraftedζAttack():
    """
    Paper: Local Model Poisoning Attacks to Byzantine-Robust Federated Learning
    code reference: https://github.com/Naiftt/SPAFD/
    """

    def __init__(self, cfg, model_at_start, device="cpu"):
        self.device = device
        self.byz_cfg = cfg.byztn_cfg
        self.lmbd = cfg.byztn_cfg["lambda"] # 0.1 in paper

        self.vec_state_ignore = ["num_batches_tracked"]

        self.lmbd = self.lmbd + (random.random()-0.5)*0.1
        print("ATTACK: FangCraftedζAttack", "Lambda", self.lmbd)


    def modify(self, lmodel_state_tth, gmodel_state_tminus1, omniscience={}):
        local_states = []
        for kid in omniscience.keys(): # clientwise train info
            local_states.append(omniscience[kid]["model"].state_dict())
        benign_state = fedops.global_average_statedict(local_states)

        wvec_benign = fedops.get_param_from_state(copy.deepcopy(benign_state),
                                keys_to_ignore=self.vec_state_ignore).to(self.device)

        gwvec_tminus1 = fedops.get_param_from_state(gmodel_state_tminus1,
                                keys_to_ignore=self.vec_state_ignore).to(self.device)
        S = (wvec_benign > gwvec_tminus1).long()
        S[S==0] = -1
        attack_wvec =  wvec_benign - (self.lmbd*S)

        out_state = fedops.set_param_in_state(lmodel_state_tth, attack_wvec,
                                    keys_to_ignore=self.vec_state_ignore)

        return out_state


class ALIEζAttack():
    """
    Paper: Baruch - A Little Is Enough: Circumventing Defenses For Distributed Learning
    code modified: https://github.com/epfml/byzantine-robust-optimizer/tree/main/codes/attacks
    """

    def __init__(self, cfg, model_at_start, device="cpu"):

        self.device = device
        self.byz_cfg = cfg.byztn_cfg
        self.z_max = self.byz_cfg.get("z_max")

        self.num_client_k = n = cfg["data_centers_count"]
        self.num_byzant_b = m = len(self.byz_cfg["byztn_clients"])
        self.num_honest_g = g = n-m

        self.vec_state_ignore = ["num_batches_tracked"]

        ## this is global common start point
        self.gwvec_0th:torch.Tensor = fedops.get_param_from_state(model_at_start.state_dict(),
                                    keys_to_ignore=self.vec_state_ignore)

        if not self.z_max:
            s = np.floor(n / 2 + 1) - m
            cdf_value = (n - m - s) / (n - m)
            self.z_max = spstats.norm.ppf(cdf_value)

        self.z_max = self.z_max + (random.random()-0.5)*0.1
        print("ATTACK: ALIEζAttack", "Z:", self.z_max)

    def modify(self, lmodel_state_tth, gmodel_state_tminus1, omniscience={}):

        # Loop over benign weights
        benign_wvec_list = []
        for kid in omniscience.keys(): # clientwise train info
            benign_wvec_list.append( fedops.get_param_from_state(
                    omniscience[kid]["model"].state_dict(),
                    keys_to_ignore=self.vec_state_ignore).to(self.device) )
        benign_wvec = torch.vstack(benign_wvec_list)

        gwvec_tminus1 = fedops.get_param_from_state(
                    gmodel_state_tminus1,
                    keys_to_ignore=self.vec_state_ignore).to(self.device)

        benign_grads = gwvec_tminus1 - benign_wvec   ## ΔW

        mu = torch.mean(benign_grads, dim=0)
        std = torch.std(benign_grads, dim=0)
        attack_grad = mu - std * self.z_max

        attack_wvec = gwvec_tminus1 - attack_grad # W - ΔW

        out_state = fedops.set_param_in_state(copy.deepcopy(lmodel_state_tth), attack_wvec,
                                    keys_to_ignore=self.vec_state_ignore)
        return out_state


class XieIPMζAttack():
    """
    Paper: Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation
    code modified: https://github.com/epfml/byzantine-robust-optimizer/tree/main/codes/attacks
    """
    def __init__(self, cfg, model_at_start, device="cpu"):

        self.device = device
        self.byz_cfg = cfg.byztn_cfg
        self.epsilon = self.byz_cfg.get("epsilon")

        self.num_client_k = n = cfg["data_centers_count"]
        self.num_byzant_b = m = len(self.byz_cfg["byztn_clients"])
        self.num_honest_g = g = n-m

        self.vec_state_ignore = ["num_batches_tracked"]

        ## this is global common start point
        self.gwvec_0th:torch.Tensor = fedops.get_param_from_state(model_at_start.state_dict(),
                                    keys_to_ignore=self.vec_state_ignore)

        self.epsilon = self.epsilon + (random.random()-0.5)*0.1
        print("ATTACK: XieIPMζAttack")


    def modify(self, lmodel_state_tth, gmodel_state_tminus1, omniscience={}):

        # Loop over benign weights
        benign_wvec_list = []
        for kid in omniscience.keys(): # clientwise train info
            benign_wvec_list.append( fedops.get_param_from_state(
                    omniscience[kid]["model"].state_dict(),
                    keys_to_ignore=self.vec_state_ignore).to(self.device) )
        benign_wvec = torch.vstack(benign_wvec_list)

        gwvec_tminus1 = fedops.get_param_from_state(gmodel_state_tminus1,
                                    keys_to_ignore=self.vec_state_ignore)

        delta_wvec = gwvec_tminus1 - benign_wvec # ΔW
        # Wt = Wt-1 - ε(-ΔW)
        attack_wvec = gwvec_tminus1 + self.epsilon * (torch.mean(delta_wvec, dim=0))

        out_state = fedops.set_param_in_state(copy.deepcopy(lmodel_state_tth), attack_wvec,
                                    keys_to_ignore=self.vec_state_ignore)
        return out_state



class MimicζAttack():
    """
    Paper: Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing
    code modified: https://github.com/epfml/byzantine-robust-noniid-optimizer/tree/main/codes/attacks
    """
    def __init__(self, cfg, model_at_start, device="cpu"):

        self.device = device
        self.byz_cfg = cfg.byztn_cfg
        self.warmup_steps = self.byz_cfg.get("warmup_steps")

        self.num_client_k = n = cfg["data_centers_count"]
        self.num_byzant_b = m = len(self.byz_cfg["byztn_clients"])
        self.num_honest_g = g = n-m

        self.byzant_ranks = self.byz_cfg["byztn_clients"]
        self.honest_ranks = list( set(range(self.num_client_k)) - set(self.byzant_ranks) )

        self.vec_state_ignore = ["num_batches_tracked"]
        ## this is global common start point
        self.gwvec_0th:torch.Tensor = fedops.get_param_from_state(model_at_start.state_dict(),
                                    keys_to_ignore=self.vec_state_ignore)

        self.t  = 0
        self.target_rank = None
        self.mu = torch.zeros_like(self.gwvec_0th, device=self.device)

        gen = torch.Generator(device=self.device)
        gen.manual_seed(0)
        self.z = torch.rand(self.gwvec_0th.shape, generator=gen, device=self.device)

        print("ATTACK: MimicζAttack")


    def _warmup_routine(self, curr_good_gradvecs:torch.tensor):
        #NOTE: Paper seems to be okay with Weights yet implementation had gradients
        # weights are passed instead of grad

        curr_gr = curr_good_gradvecs
        curr_gr_avg = curr_good_gradvecs.mean(dim=0)

        ### Finding the Zee

        self.mu = self.t / (1 + self.t) * self.mu + curr_gr_avg / (1 + self.t)

        cumul = ( (curr_gr - self.mu)*(curr_gr - self.mu) ).sum(dim=0)
        self.z = (self.t/(1+self.t)) * self.z + \
                (cumul/cumul.norm() / (1+self.t) ) * self.z
        self.z = self.z / self.z.norm()

        ### client to mimic
        g_dot_z = (curr_gr *self.z).sum(dim=1)
        mv = torch.max(g_dot_z)
        mi = torch.argmax(g_dot_z)
        mg = curr_gr[mi]
        print("ζ"*5 +"Warmup Mimic Client", mi, "val", mv)

        return mi

    def modify(self, lmodel_state_tth, gmodel_state_tminus1, omniscience={}):
        good_wvec_list = []
        for kid in omniscience.keys(): # clientwise train info
            if kid in self.honest_ranks:
                good_wvec_list.append( fedops.get_param_from_state(
                        omniscience[kid]["model"].state_dict(),
                        keys_to_ignore=self.vec_state_ignore).to(self.device) )
        stacked_good_wvec = torch.vstack(good_wvec_list)

        # Figure out the client to mimic
        if (self.t < self.warmup_steps) or (self.target_rank is None):
            self.target_rank = self._warmup_routine(stacked_good_wvec)
        else: # Fixed that client
            self.target_rank = self.target_rank

        print("Mimicing", self.target_rank)
        self.t += 1

        attack_wvec = stacked_good_wvec[self.target_rank]
        out_state = fedops.set_param_in_state(lmodel_state_tth, attack_wvec,
                                    keys_to_ignore=self.vec_state_ignore)

        return out_state



## TODO: Fix the byzantine tolerance
class OzfaturaROPζAttack():
    """
    Reference: Byzantines can also Learn from History: Fall of Centered Clipping in Federated Learning
    modified on basecode from authors
    """

    def __init__(self, cfg, model_at_start, device="cpu"):
        self.device = device
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
        self.gwvec_tminus2:torch.Tensor = fedops.get_param_from_state(model_at_start.state_dict(),
                                    keys_to_ignore=self.vec_state_ignore)

        if not self.z_max:
            s = np.floor(n / 2 + 1) - m
            cdf_value = (n - m - s) / (n - m)
            self.z_max = spstats.norm.ppf(cdf_value)

        self.z_max = self.z_max + (random.random()-0.5)*0.1
        print("ATTACK: OzfaturaROPζAttack", "Z", self.z_max)

    def modify(self, lmodel_state_tth, gmodel_state_tminus1, omniscience={}):

        local_states = []
        for kid in omniscience.keys(): # clientwise train info
            local_states.append(omniscience[kid]["model"].state_dict())
        benign_state = fedops.global_average_statedict(local_states)
        benign_wvec = fedops.get_param_from_state(copy.deepcopy(benign_state),
                                    keys_to_ignore=self.vec_state_ignore).to(self.device)

        gwvec_tminus1 = fedops.get_param_from_state(
                    gmodel_state_tminus1,
                    keys_to_ignore=self.vec_state_ignore).to(self.device)

        benign_grads = gwvec_tminus1 - benign_wvec

        ## m_bar_t :-> aggregate of Benign Gradients
        m_bar_t = benign_grads

        ## Global aggregate of all m~(t-1) :-> ud = self.global_momentum.clone()
        m_tilde_tminus1 =  self.gwvec_tminus2 - gwvec_tminus1

        ### reference point, global momentum
        # here we can take 0th or (t-1)th based on clipping defense used
        # if first iteration, set the reference point to the mean of the benign momentums
        if self.first_step_ignore:
            m_tilde_tminus1 = m_bar_t.clone()
            self.first_step_ignore = False

        ## Target point of attack ; between previous and current
        # m_hat_t :-> ud
        ud = (m_tilde_tminus1 * self.lmbd) + (m_bar_t * (1-self.lmbd))

        ## Orthogonal vector
        pert = torch.ones_like(m_bar_t)  # inital perturbation
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
        attack_location = (m_tilde_tminus1 * self.rho) + (m_bar_t * (1-self.rho)) #relocate reference
        attack_grad = attack_location + pert # final attack added to desired location

        attack_wvec = gwvec_tminus1 - attack_grad
        out_state = fedops.set_param_in_state(copy.deepcopy(lmodel_state_tth), attack_wvec,
                                    keys_to_ignore=self.vec_state_ignore)

        del self.gwvec_tminus2
        self.gwvec_tminus2 = gwvec_tminus1.clone()

        return out_state
