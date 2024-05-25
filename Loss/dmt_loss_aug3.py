# a pytorch based lisv2 code

# from multiprocessing import Pool

import numpy as np
import torch
import torch.autograd
# import torch.nn.functional as F
# from scipy import optimize
from torch import nn
# from torch.autograd import Variable
# from torch.functional import split
# from torch.nn.modules import loss
# from typing import Any
import scipy

# from Loss.dmt_loss_source import Source


def UMAPNoSigmaSimilarity(dist, gamma, v=100, h=1, pow=2):

    dist_rho = dist

    dist_rho[dist_rho < 0] = 0
    Pij = (
        gamma
        * torch.tensor(2 * 3.14)
        * gamma
        * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1))
    )
    return Pij


class MyLoss(nn.Module):
    def __init__(
        self,
        v_input,
        SimilarityFunc=UMAPNoSigmaSimilarity,
        metric="braycurtis",
        eta=1,
        near_bound=0,
        far_bound=1,
        augNearRate=10000,
    ):
        super(MyLoss, self).__init__()

        self.v_input = v_input
        self.gamma_input = self._CalGamma(v_input)
        self.ITEM_loss = self._TwowaydivergenceLoss
        self._Similarity = SimilarityFunc
        self.metric = metric
        self.eta = eta
        self.near_bound = near_bound
        self.far_bound = far_bound
        self.augNearRate = augNearRate

    def forward(
        self,
        input_data1,
        input_data2,
        latent_data1,
        latent_data2,
        v_latent,
        input_tp=None,
        latent_tp=None,
        metric='euclidean',
    ):
        # data_1 = input_data[: input_data.shape[0] // 2]
        dis_data_tp1 = self._DistanceSquared(input_tp, input_data1, metric=metric)
        dis_data_tp2 = self._DistanceSquared(input_tp, input_data2, metric=metric)
        Sim_data_tp1 = self._Similarity(dist=dis_data_tp1,gamma=self.gamma_input,v=self.v_input, )
        Sim_data_tp2 = self._Similarity(dist=dis_data_tp2,gamma=self.gamma_input,v=self.v_input, )
        
        dis_late_tp1 = self._DistanceSquared(latent_tp, latent_data1)
        dis_late_tp2 = self._DistanceSquared(latent_tp, latent_data2)
        Sim_late_tp1 = self._Similarity(dist=dis_late_tp1, gamma=self._CalGamma(v_latent),v=v_latent, )
        Sim_late_tp2 = self._Similarity(dist=dis_late_tp2, gamma=self._CalGamma(v_latent),v=v_latent, )

        loss_ce_1 = self.ITEM_loss(P_=Sim_data_tp1, Q_=Sim_late_tp2)
        loss_ce_2 = self.ITEM_loss(P_=Sim_data_tp2, Q_=Sim_late_tp1)

        return loss_ce_1+loss_ce_2

    def ForwardInfo(
        self,
        input_data,
        latent_data,
        rho,
        sigma,
        v_latent,
    ):

        dis_P = self._DistanceSquared(input_data)
        P = self._Similarity(
            dist=dis_P,
            rho=rho,
            sigma_array=sigma,
            gamma=self.gamma_input,
            v=self.v_input,
        )

        dis_Q = self._DistanceSquared(latent_data)
        Q = self._Similarity(
            dist=dis_Q, rho=0, sigma_array=1, gamma=self._CalGamma(v_latent), v=v_latent
        )

        loss_ce = self.ITEM_loss(
            P_=P,
            Q_=Q,
        )
        return (
            loss_ce.detach().cpu().numpy(),
            dis_P.detach().cpu().numpy(),
            dis_Q.detach().cpu().numpy(),
            P.detach().cpu().numpy(),
            Q.detach().cpu().numpy(),
        )

    def _TwowaydivergenceLoss(self, P_, Q_, select=None):

        EPS = 1e-5
        # select = (P_ > Q_) & (P_ > self.near_bound)
        # select_index_far = (P_ < Q_) & (P_ < self.far_bound)
        # P_ = P_[torch.eye(P_.shape[0])==0]*(1-2*EPS) + EPS
        # Q_ = Q_[torch.eye(P_.shape[0])==0]*(1-2*EPS) + EPS
        losssum1 = P_ * torch.log(Q_ + EPS)
        losssum2 = (1 - P_) * torch.log(1 - Q_ + EPS)
        losssum = -1 * (losssum1 + losssum2)

        # if select is not None:
        #     losssum = losssum[select]

        return losssum.mean()

    def _L2Loss(self, P, Q):

        losssum = torch.norm(P - Q, p=2) / P.shape[0]
        return losssum

    def _L3Loss(self, P, Q):

        losssum = torch.norm(P - Q, p=3) / P.shape[0]
        return losssum

    def _DistanceSquared(self, x, y=None, metric="euclidean"):
        if metric == "euclidean":
            if y is not None:
                m, n = x.size(0), y.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
            else:
                m, n = x.size(0), x.size(0)
                xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
                yy = xx.t()
                dist = xx + yy
                dist = torch.addmm(dist, mat1=x, mat2=x.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
                dist[torch.eye(dist.shape[0]) == 1] = 1e-12
        
        if metric == "cossim":
            input_a, input_b = x, x
            normalized_input_a = torch.nn.functional.normalize(input_a)  
            normalized_input_b = torch.nn.functional.normalize(input_b)
            dist = torch.mm(normalized_input_a, normalized_input_b.T)
            dist *= -1 # 1-dist without copy
            dist += 1

            dist[torch.eye(dist.shape[0]) == 1] = 1e-12

        return dist

    def _CalGamma(self, v):

        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out
