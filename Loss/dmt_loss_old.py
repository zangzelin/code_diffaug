from multiprocessing import Pool

import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F
from scipy import optimize
from torch import nn
from torch.autograd import Variable
from torch.functional import split
from torch.nn.modules import loss
from typing import Any
import scipy

# from Loss.dmt_loss_source import Source

def Similarity(dist, rho, sigma_array, gamma, v=100):


    dist_rho = (dist - rho) / sigma_array
    
    dist_rho[dist_rho < 0] = 0
    # Pij = torch.pow(
    #     gamma * torch.pow(
    #         (1 + dist_rho / v),
    #         -1 * (v + 1) / 2
    #         ) * torch.sqrt(torch.tensor(2 * 3.14)),
    #         2
    #     )
    Pij = gamma*gamma * torch.pow(
            (1 + dist_rho / v),
            -1 * (v + 1)
            ) * 2 * 3.14
    # print(Pij, Pij2)
    # input()
    P = Pij + Pij.t() - torch.mul(Pij, Pij.t())

    return P

class Source(nn.Module):
    def __init__(
        self,
        v_input=100,
    ):
        super(Source, self).__init__()

        # self.device = device
        self.v_input = v_input
        self.gamma_input = self._CalGamma(v_input)
        self.ITEM_loss = self._TwowaydivergenceLoss
        self._Similarity = Similarity
    
    def forward(self, input_data, latent_data, rho, sigma, v_latent):
        
        
        loss_ce = self.ITEM_loss(
            P=self._Similarity(
                dist=self._DistanceSquared(input_data),
                rho=rho,
                sigma_array=sigma,
                gamma=self.gamma_input,
                v=self.v_input),
            Q=self._Similarity(
                dist=self._DistanceSquared(latent_data),
                rho=0,
                sigma_array=1,
                gamma=self._CalGamma(v_latent),
                v=v_latent)
        )
        
        return loss_ce

    def _TwowaydivergenceLoss(self, P, Q):

        EPS = 1e-12
        P_ = P[torch.eye(P.shape[0])==0]*(1-2*EPS) + EPS
        Q_ = Q[torch.eye(P.shape[0])==0]*(1-2*EPS) + EPS
        losssum1 = (P_ * torch.log(Q_ + EPS)).mean()
        losssum2 = ((1-P_) * torch.log(1-Q_ + EPS)).mean()
        losssum = -1*(losssum1 + losssum2)

        if torch.isnan(losssum):
            input('stop and find nan')
        return losssum

    def _L2Loss(self, P, Q):

        losssum = torch.norm(P-Q, p=2)/P.shape[0]
        return losssum
    
    def _L3Loss(self, P, Q):

        losssum = torch.norm(P-Q, p=3)/P.shape[0]
        return losssum

    # def _Similarity(self, dist, rho, sigma_array, gamma, v=100):

    #     if torch.is_tensor(rho):
    #         dist_rho = (dist - rho) / sigma_array
    #     else:
    #         dist_rho = dist
        
    #     dist_rho[dist_rho < 0] = 0
    #     Pij = torch.pow(
    #         gamma * torch.pow(
    #             (1 + dist_rho / v),
    #             -1 * (v + 1) / 2
    #             ) * torch.sqrt(torch.tensor(2 * 3.14)),
    #             2
    #         )

    #     P = Pij + Pij.t() - torch.mul(Pij, Pij.t())

    #     return P
    
    def _DistanceSquared(
        self,
        x,
    ):
        m, n = x.size(0), x.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = xx.t()
        dist = xx + yy
        dist.addmm_(1, -2, x, x.t())
        dist = dist.clamp(min=1e-12)
        # d[torch.eye(d.shape[0]) == 1] = 1e-12

        return dist

    def _CalGamma(self, v):
        
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out

class MyLoss(Source):
    def __init__(
        self,
        v_input=100,
        SimilarityFunc=Similarity,
        # device: Any,
    ):
        super(MyLoss, self).__init__(
            v_input,
            )

        self.ITEM_loss = self._TwowaydivergenceLoss
    
    def _TwowaydivergenceLoss(self, P, Q):

        EPS = 1e-12
        P_ = P[torch.eye(P.shape[0])==0]*(1-2*EPS) + EPS
        Q_ = Q[torch.eye(P.shape[0])==0]*(1-2*EPS) + EPS
        losssum1 = (P_ * torch.log(Q_ + EPS)).mean()
        losssum2 = ((1-P_) * torch.log(1-Q_ + EPS)).mean()
        losssum = -1*(losssum1 + losssum2)

        if torch.isnan(losssum):
            input('stop and find nan')
        return losssum