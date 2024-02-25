import copy
import h5py

import torch
import torch.nn.functional as torch_F
import numpy as np
import scipy
import pyod
import algorithms.federation_ops as fedops
from algorithms.federation_byz import NoGuardΞByzantine, NoGuardΞByzantineDecopl, torch_insert_diagonal, torch_remove_diagonal
import algorithms.byzantine_attacks as byz_attacks



## Decoupled weightage based on Sinkhorn distance

class WeighOmegaSKDHΞByzantineDecopl(NoGuardΞByzantineDecopl): # Attempt 0

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        self.aggregator_func = self.__dataweightage_aggregation_decopld

        with h5py.File(self.defense_cfg["datasummary"], 'r') as hdf5_file:
            data_dist = hdf5_file["dist_matrix"][()]
            data_dist = data_dist[:-1, :-1] # ignore all distances

        self.client_weightage = self._preset_weightage(data_dist)

        print("Defense: Decoupled WeighOmega-SKHD")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------

    def _preset_weightage(self, data_dist):
        K = data_dist.shape[0]

        ## FOR ISIC dataset
        # matx = torch.tensor([0.53,0.17,0.14,0.10,0.04,0.02]*K).view(K,K)

        matx = torch.ones((K,K)) * (1/K)

        return matx

    def _distance_naive_softmin_weightage(self, data_dist):
        data_dist = (data_dist + data_dist.T) / 2

        data_dist = torch.tensor(data_dist)

        normed_dist = (data_dist - data_dist.min(dim=1, keepdim=True)[0]) /  \
                    (data_dist.max(dim=1, keepdim=True)[0] - data_dist.min(dim=1, keepdim=True)[0])

        weightage_matrix = torch_F.softmin(torch.tensor(normed_dist), dim=1)
        return weightage_matrix

    def _mena_soft_weightage(self, data_dist):
        data_dist = (data_dist + data_dist.T) / 2
        data_dist = torch.tensor(data_dist)

        K = data_dist.shape[0] #clients

        rmean_dist = torch.mean(data_dist, dim=0)

        all_dist =  (data_dist +
            rmean_dist * torch.eye(K, dtype=float).to_dense())

        rel_dist = all_dist / rmean_dist.view(-1,1)
        rel_dist = torch.clamp(rel_dist, min=1.0)

        weightage_matrix = torch_F.softmin(rel_dist)

        return weightage_matrix


    def _boltzman_factor_weightage(self, data_dist):
        """ Follows Boltzman Distribution paradigm
        data_dist: numpy arr
        return : torch.tensor
        """
        data_dist = (data_dist + data_dist.T) / 2
        # data_dist = data_dist.T
        K = data_dist.shape[0]

        data_dist = torch.tensor(data_dist)

        ## 90th - 1.282 | 95th - 1.645 | 99th - 2.326 | 75th - 0.674
        sigz = 1.645

        ## COMPUTE Temperature
        non_diag = torch_remove_diagonal(data_dist)
        dist_mu = torch.mean(non_diag)
        sigma = torch.std(non_diag)

        dist_max, dist_min = dist_mu+sigz*sigma, dist_mu-sigz*sigma
        tempK = (dist_max - dist_min) / (np.log(1/6) - np.log(5/6))

        ### COMPUTE beta_ij
        softin_beta = non_diag / tempK
        beta_ij = torch.softmax(softin_beta, dim=1)

        ## randomize compute betas along row
        # perm = torch.randperm(beta_ij.shape[1])
        # beta_ij = beta_ij[:, perm]

        ## constant betas
        # beta_ij = beta_ij*0 + (1/5)

        beta_ij = torch_insert_diagonal(beta_ij)

        ### COMPUTE alpha_ii
        # drowmean = (data_dist.sum(dim=0)/(data_dist.shape[0]-1))
        # softin_alpha = drowmean / tempK
        # alpha_ii = torch.softmax(softin_alpha, dim=0)
        # alpha_ii = torch.max(beta_ij, dim=1)[0]

        ## constant alpha
        alpha_ii = 1.5/ data_dist.shape[0] #approx 0.25 for isic

        # omega_ij full
        weightage_matrix = ((1-alpha_ii) * beta_ij +
                        alpha_ii * torch.eye(data_dist.shape[0], dtype=float).to_dense())

        return weightage_matrix

    #-------- Server methods ----------

    def __dataweightage_aggregation_decopld(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])

        # fully_wvecs = [fedops.get_param_from_state(l["model_state"])
        #             for l in lsets]
        # sfully_wvec = torch.vstack(fully_wvecs)

        wvecs = [fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore = self.vec_state_ignore)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)

        out_ref_wvec = torch.zeros_like(stacked_wvec)

        agg_states_cli = {}
        for i in range(len(lsets)):
            cweigh = self.client_weightage[i,:].view(-1, 1)
            cweighed_wvec = cweigh.to(self.device) * stacked_wvec  # s1*[v1] \ s2*[v2] \ s3*v3 ...
            cli_wvec = torch.sum(cweighed_wvec, dim = 0)
            agg_states = fedops.set_param_in_state(state_dict_struct, cli_wvec)
            agg_states_cli.update({f"client_{i}": copy.deepcopy(agg_states)})
            out_ref_wvec[i, :] = cli_wvec

        agg_states = fedops.set_param_in_state(state_dict_struct, out_ref_wvec.mean(dim=0),
                                               keys_to_ignore=self.vec_state_ignore)
        agg_states_cli.update({"client_G": copy.deepcopy(agg_states)})

        info_dict = {"client_weightage":self.client_weightage.tolist()}
        return agg_states_cli, info_dict


##------------------------------------------------------------------------------


class TauThetaLambdaSKDHΞByzantineDecopl(NoGuardΞByzantineDecopl): # Attempt 1

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        self.aggregator_func = self.__dynamic_Tau_Theta_Lambda_aggr_decopld

        with h5py.File(self.defense_cfg["datasummary"], 'r') as hdf5_file:
            data_dist = hdf5_file["dist_matrix"][()]
            data_dist = data_dist[:-1, :-1] # ignore all distances

        self.featx_d  = cfg.defense_cfg["feature_extractor_d"]  # D --> final feature size before classifier
        self.client_clip_factor = self._get_lmbda_from_dist(data_dist)

        ##
        self.vec_state_ignore = ["num_batches_tracked"] # critical for l2norms since this skews it
        if self.id == "G": #large tensors, so why waste mem
            self.init_stacked_wvecs()

        print("Defense: Decoupled Clipping Tau-SKHD")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------

    def init_stacked_wvecs(self):
        wvec = fedops.get_param_from_state(self.gmodel_init.state_dict(),
                        keys_to_ignore=self.vec_state_ignore)
        self.wvec_init = wvec.clone()
        self.aggwvec_tminus1 = torch.vstack( [wvec]*self.num_client_k )
        self.wvec_tminus1    = self.aggwvec_tminus1.clone()
        # fully_wvec =  fedops.get_param_from_state(self.gmodel_init.state_dict())
        # self.fully_aggwvec_tminus1 = torch.vstack( [fully_wvec]*self.num_client_k )


    def _get_constant_tau(self, data_dist):
        K = data_dist.shape[0]
        tau_val = 1.0
        matx = torch.ones((K,K)) * tau_val
        return matx


    def _get_lmbda_from_dist(self, data_dist):
        data_dist = (data_dist + data_dist.T) / 2
        data_dist = torch.tensor(data_dist)
        K = data_dist.shape[0] #clients
        D = self.featx_d

        # rmean_dist = torch.sum(data_dist, dim=0) / K
        # all_dist = data_dist + (rmean_dist * torch.eye(K, dtype=float).to_dense())

        own_dist = torch.diag(data_dist)

        lambda_dist = torch.div(data_dist, own_dist) # scale the distance
        lambda_dist = 1 + torch.abs(1 - lambda_dist) # mirror the deviation

        lambda_dist = lambda_dist * (1 - torch.eye(K, K)) # make diagonal zeros

        return lambda_dist.to(torch.float)


    #-------- Server methods ----------
    def safe_divide(self, nu, de, fill=1.0):
        res = torch.full_like(de, fill_value=fill)
        mask = (de != 0.0)

        if (nu.shape == mask.shape): nu_ = nu[mask]
        elif (sum(nu.shape) == 1):   nu_ = nu
        else: raise Exception(f"Incompatible shapes {de.shape}, {nu.shape}")

        res[mask] = torch.div(nu_, de[mask])
        return res


    def __dynamic_Tau_Theta_Lambda_aggr_decopld(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])

        #for fedavging with bn_batches_tracked; thanks to Pytorch default models for complicating life
        # fully_wvecs = [fedops.get_param_from_state(l["model_state"])
        #             for l in lsets]
        # sfully_wvec = torch.vstack(fully_wvecs)
        # outfully_aggwvec = torch.zeros_like(sfully_wvec)
        # sfully_aggwvec_tminus1 = self.fully_aggwvec_tminus1

        #for l2norms
        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)
        outref_aggwvec = torch.zeros_like(stacked_wvec)

        stacked_wvec_tminus1    = self.wvec_tminus1
        stacked_aggwvec_tminus1 = self.aggwvec_tminus1

        ##
        print(torch.norm(stacked_aggwvec_tminus1 - stacked_wvec[0], dim=1).view(-1, 1))

        ## Lambda Computes
        lmbda_dist = self.client_clip_factor.to(self.device)
        ## Tau Computes
        tau_rad = torch.norm(stacked_aggwvec_tminus1 - stacked_wvec, dim=1).view(-1, 1)
        ## Theta Computes
        theta_angs = torch.acos(torch_F.cosine_similarity(
                        stacked_aggwvec_tminus1, stacked_wvec, dim=1).view(-1, 1))

        agg_states_cli = {}
        sector_scales = []; rad_scales = []; cos_scales = []
        for i in range(stacked_wvec.shape[0]):
            taui = (tau_rad[i]* lmbda_dist[i]).view(-1, 1)
            ccden = torch.norm(stacked_wvec[i] - stacked_wvec, dim=1).view(-1,1)
            taui_by_ccden = self.safe_divide(taui, ccden, fill=1.0) # this will return 1 for taui by ccdenii
            rad_comp = torch.minimum(torch.tensor(1), taui_by_ccden).view(-1,1)

            costhetai = torch.cos(theta_angs[i])#Strict #* lmbda_dist[i]).view(-1, 1)
            cosbase = torch_F.cosine_similarity(stacked_wvec[i], stacked_wvec, dim=1).view(-1,1)
            cosbase_x_thetai = costhetai*100*(cosbase-costhetai)
            cos_comp = torch.clamp(2*torch_F.sigmoid(cosbase_x_thetai), min=0.0, max=1.0)


            scale_sec = rad_comp * cos_comp #* lmbda_dist[i].view(-1,1)
            scale_sec = torch.clamp(scale_sec, max=1, min=0)
            # print("Scale Shape", scale_sec.shape)

            ## start core
            clipped_deltawvec = scale_sec * (stacked_wvec - stacked_aggwvec_tminus1) # s1*[v1] \ s2*[v2] \ s3*v3 ...
            clipped_aggdeltawvec = \
                torch.sum(clipped_deltawvec, dim = 0) / torch.sum(scale_sec)

            client_wvec = stacked_aggwvec_tminus1[i] + clipped_aggdeltawvec
            outref_aggwvec[i, :] = client_wvec

            sector_scales.append(scale_sec.flatten().tolist())
            rad_scales.append(rad_comp.flatten().tolist())
            cos_scales.append(cos_comp.flatten().tolist())

            agg_states = fedops.set_param_in_state(state_dict_struct, client_wvec,
                                            keys_to_ignore=self.vec_state_ignore)
            agg_states_cli.update({f"client_{i}": copy.deepcopy(agg_states)})
            ## end core

        agg_states = fedops.set_param_in_state(state_dict_struct, outref_aggwvec.mean(dim=0),
                                               keys_to_ignore=self.vec_state_ignore)
        agg_states_cli.update({"client_G": copy.deepcopy(agg_states)})

        self.wvec_tminus1    = stacked_wvec.clone()
        self.aggwvec_tminus1 = outref_aggwvec.clone()

        info_dict = {"client_clip_weightage":sector_scales,
                     "radius_component": rad_scales,
                     "cosine_component": cos_scales}

        return agg_states_cli, info_dict


##==============================================================================

class NewTauLambdaΞByzantine(NoGuardΞByzantine): # Attempt 2

    def __init__(self, cfg, id, model, device="cpu"):
        self.id = id
        self.device = device
        self.byz_way = None
        self.cfg = cfg
        self.byztn_cfg = cfg.byztn_cfg
        self.defense_cfg = cfg.defense_cfg
        self.num_client_k = K = int(cfg.data_centers_count) # K

        self.gmodel_init = copy.deepcopy(model).to(self.device) # model recieved at start
        self.gmodel_tminus1  = copy.deepcopy(model).to(self.device) # model recieved at Tth global comm
        self.vec_state_ignore = ["num_batches_tracked"]

        self.aggregator_func = self.__dynamic_Tau_aggregate

        with h5py.File(self.defense_cfg["datasummary"], 'r') as hdf5_file:
            data_dist = hdf5_file["dist_matrix"][()]
            data_dist = data_dist[:K, :K] # ignore all distances row-column

        self.featx_d  = cfg.defense_cfg["feature_extractor_d"]  # D --> final feature size before classifier
        self.lambda_factor = self._get_lmbda_from_dist(data_dist)


        ##
        self.vec_state_ignore = ["num_batches_tracked"] # critical for l2norms since this skews it
        if self.id == "G": #large tensors, so why waste mem
            self.init_stacked_wvecs(model)

        print("Defense: Fellowship of the Ring")

        if len(self.byztn_cfg) != 0:
            self._init_byzantiness()

    ##--------------------

    def init_stacked_wvecs(self, model):
        wvec = fedops.get_param_from_state(model.state_dict(),
                        keys_to_ignore=self.vec_state_ignore)
        self.wvec_init = wvec.clone()
        self.aggwvec_tminus1 = wvec.clone()


    def _get_constant_tau(self, data_dist):
        K = data_dist.shape[0]
        tau_val = 1.0
        matx = torch.ones((K,K)) * tau_val
        return matx


    def _get_lmbda_from_dist(self, data_dist):
        data_dist = (data_dist + data_dist.T) / 2
        data_dist = torch.tensor(data_dist)
        K = data_dist.shape[0] #clients
        D = torch.tensor(self.featx_d)

        lambda_dist = torch.div(data_dist, torch.sqrt(D)) # scale the distance
        lambda_dist = 1 + torch.abs(1 - lambda_dist) # mirror the deviation

        return lambda_dist.to(torch.float)


    #-------- Server methods ----------

    def safe_divide(self, nu, de, fill=0.0):
        res = torch.full_like(de, fill_value=fill)
        mask = (de != 0.0)

        if (nu.shape == mask.shape): nu_ = nu[mask]
        elif (sum(nu.shape) == 1):   nu_ = nu
        else: raise Exception(f"Incompatible shapes {de.shape}, {nu.shape}")

        res[mask] = torch.div(nu_, de[mask])
        return res


    def __dynamic_Tau_aggregate(self, lsets):
        state_dict_struct = copy.deepcopy(lsets[0]["model_state"])

        #for l2norms
        wvecs =[fedops.get_param_from_state(l["model_state"],
                    keys_to_ignore=self.vec_state_ignore)
                    for l in lsets]
        stacked_wvec = torch.vstack(wvecs)
        stacked_deltawvec = torch.zeros_like(stacked_wvec)

        aggwvec_tminus1 = self.aggwvec_tminus1

        K = len(lsets)

        ## Lambda Computes
        lmbd_rel = self.lambda_factor.to(self.device)
        lmbd_ij = lmbd_rel * (1 - torch.eye(K, K).to(self.device)) # make diagonal zeros since i vs i true distance is zero
        lmbdorth_ii = torch.diag(lmbd_rel).view(K, 1) # get distance between orthogonal distance
        lmbd_mat = torch.minimum(lmbdorth_ii, lmbd_ij)

        rad_scales = []; g_deltas = []
        for i in range(stacked_wvec.shape[0]):
            ## Tau Computes
            gdelta = torch.norm(aggwvec_tminus1 - stacked_wvec[i], dim=1).view(1) #scalar
            taui = lmbd_mat[i].view(-1, 1) * gdelta

            ccden = torch.norm(stacked_wvec - stacked_wvec[i], dim=1).view(-1,1)
            taui_by_ccden = self.safe_divide(taui, ccden, fill=0.0) ## fills i by i as zero w.r.t method of interest
            rad_comp = torch.minimum(torch.tensor(1), taui_by_ccden).view(-1,1)

            rad_comp = taui *0 +1 #^^^^^)))))))) Bypasss

            ## start core
            clipped_deltawvec = rad_comp * (stacked_wvec - aggwvec_tminus1) # s1*[v1] \ s2*[v2] \ s3*v3 ...
            clipped_aggdelta_i = torch.sum(clipped_deltawvec, dim = 0) / (K-1)

            stacked_deltawvec[i, :] = clipped_aggdelta_i

            rad_scales.append(rad_comp.flatten().tolist())
            g_deltas.append(gdelta.item())
            ## end core

        out_wvec  = aggwvec_tminus1 + (torch.sum(stacked_deltawvec, dim=0) / K)
        agg_state = fedops.set_param_in_state(state_dict_struct, out_wvec,
                                               keys_to_ignore=self.vec_state_ignore)

        self.aggwvec_tminus2 = aggwvec_tminus1.clone()
        self.aggwvec_tminus1 = out_wvec.clone()

        info_dict = {"client_clip_weightage":rad_scales, "g_delta": g_deltas}

        return agg_state, info_dict