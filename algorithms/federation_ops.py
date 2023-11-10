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
    def process_local(zxs={}): #used at end of local round at each client
        """ Return: lstat
        """
        lstat = {}
        return lstat

    @staticmethod  #Global calculation to send to locals
    def aggregate_globally(lstats): #used at begining of local round central
        """ Return: aggregated stat
        """
        gstat = {}
        return gstat

    @staticmethod #
    def compute_local_deviation(bstat, aggstat, cen_id=None): #at each client
        """
        """
        loss, print_info = torch.tensor(0), {}
        return loss, print_info

##==============================================================================


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















##==============================================================================
"""

"""