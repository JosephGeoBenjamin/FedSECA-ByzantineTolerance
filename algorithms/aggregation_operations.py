# Ξ - chi used as underscore in Classes

import torch
from torch import nn


class SpecificMethodΞMainTemplate():

    @staticmethod #think Local calculations send to global
    def accumulate_locals(zxs, stat_dict={}):
        """ zxs is concatenation of all local zxs
        """
        fZX = zxs.detach().clone()
        if stat_dict:
            fZX = torch.cat([stat_dict["featZX"], fZX], axis=0)
        return {"featZX":fZX}

    @staticmethod #think Local calculations send to global
    def create_local_mvstats(zxs=0, stat_dict={}):
        """ eigne value computation
        """
        Z = stat_dict["featZX"]
        Z = torch.nn.functional.normalize(Z, dim=1)
        autocorr = torch.matmul(Z.T, Z) / Z.shape[0]
        ## original vne --> [-Z.shape[0]:] --> size of batch;
        ## here changing to  [-Z.shape[1]:] --> feature dim
        eig_val = torch.linalg.eigvalsh(autocorr.to(torch.float32))[-Z.shape[1]:]
        return {"EigVals":eig_val}


    @staticmethod  #think Global calculations send to local
    def get_global_aggregates(local_bstats):
        """ assumption eig(A+B) <= eig(A)+eig(B)
        """
        sum_eigs = torch.tensor(0)

        for stat in local_bstats:
            sum_eigs = stat["EigVals"] + sum_eigs

        eigs = sum_eigs / len(local_bstats)
        return  {"EigVals":sum_eigs} #aggbstat


    @staticmethod #local using global context for calculations
    def compute_local_loss(bstat, aggstat, cen_id=None): #single local
        """ bstat: tensor undetached from compute graph of local models
        """
        if not aggstat: return torch.tensor(0), {}

        Z = bstat["lZX"]
        device = bstat["lZX"]

        agg_eigs = aggstat["EigVals"].to(device)

        Z = torch.nn.functional.normalize(Z, dim=1)
        autocorr = torch.matmul(Z.T, Z) / Z.shape[0]
        ## original vne --> [-Z.shape[0]:] --> size of batch;
        ## here changing to  [-Z.shape[1]:] --> feature dim
        b_eigs = torch.linalg.eigvalsh(autocorr.to(torch.float32))[-Z.shape[1]:]

        ## Way 1 Forbenius
        loss = (b_eigs - agg_eigs).pow_(2).sum().sqrt()


        print_info = { "Eigen Divergence" : round(loss.item(), 4),}

        return loss, print_info
















##==============================================================================
"""

"""