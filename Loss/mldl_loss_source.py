# a pytorch based lisv2 code
# author: ***
# email: ***@gmail.com

# import pdb
# from multiprocessing import Pool

# import numpy as np
import torch
import torch.autograd
# import torch.nn.functional as F
# from scipy import optimize
from torch import nn
# from torch.autograd import Variable
# from torch.functional import split
# from torch.nn.modules import loss
from typing import Any

# import scipy


class Source(nn.Module):
    def __init__(
        self,
        device: Any,
        regular_B: float,
        chang_start: float,
        chang_end: float,
        rate_push: float,
        epsilon=None,
        K=None,
    ):
        super(Source, self).__init__()

        self.device = device
        self.kNN_data = None
        self.regular_B = regular_B
        self.chang_start = chang_start
        self.chang_end = chang_end
        self.rate_push = rate_push
        if K is None:
            self.epsilon = epsilon
            self.K = None
            self.Neighbor = self.Epsilonball
            print("use Epsilonball")
        else:
            self.K = K
            self.epsilon = None
            self.Neighbor = self.KNNGraph
            print("use KNNGraph")

    def forward(self, input_data, latent_data, dis_data, kNN_data):
        # print(kNN_data.sum())
        dis_latent, kNN_latent = self.Neighbor(
            latent_data,
        )

        loss_iso, loss_push_away = self.DistanceLoss(
            input_data,
            latent_data,
            dis_data,
            dis_latent,
            kNN_data,
            kNN_latent,
            epoch=self.epoch,
            regular_B=self.regular_B,
            chang_start=self.chang_start,
            chang_end=self.chang_end,
            rate_push=self.rate_push,
        )

        # print(loss_push_away)
        return loss_iso + loss_push_away

    def KNNGraph(self, data, k=15):

        """
        another function used to calculate the distance between point pairs and determine the neighborhood
        Arguments:
            data {tensor} -- the train data
        Outputs:
            d {tensor} -- the distance between point pairs
            kNN_mask {tensor} a mask used to determine the neighborhood of every data point
        """

        if self.K < 1:
            k = int(self.K * data.shape[0])
        else:
            k = self.K
        batch_size = data.shape[0]

        x = data.to(self.device)
        y = data.to(self.device)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # dist.addmm_(1, -2, x, y.t())
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        d = dist.clamp(min=1e-8).sqrt()  # for numerical stabili

        s_, indices = torch.sort(d, dim=1)
        indices = indices[:, : k + 1]
        kNN_mask = torch.zeros(
            (
                batch_size,
                batch_size,
            ),
            device=self.device,
        ).scatter(1, indices, 1)
        kNN_mask[torch.eye(kNN_mask.shape[0], dtype=int)] = 0

        return d, kNN_mask.bool()

    def Epsilonball(self, data):

        """
        function used to calculate the distance between point pairs and determine the neighborhood with r-ball
        Arguments:
            data {tensor} -- the train data
        Outputs:
            d {tensor} -- the distance between point pairs
            kNN_mask {tensor} a mask used to determine the neighborhood of every data point
        """

        epsilon = self.epsilon

        x = data.to(self.device)
        y = data.to(self.device)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        # dist.addmm_(1, -2, x, y.t())
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        d = dist.clamp(min=1e-8).sqrt()

        kNN_mask = (d < epsilon).bool()

        return d, kNN_mask

    def DistanceLoss(
        self,
        data,
        latent,
        dis_data,
        dis_latent,
        kNN_data,
        kNN_latent,
        epoch,
        regular_B=3,
        chang_start=500,
        chang_end=1000,
        rate_push=5,
    ):

        """
        function used to calculate loss_iso and loss_push-away
        Arguments:
            data {tensor} -- the data for input layer data
            latent {tensor} -- the data for latent layer data
            dis_data {tensor} -- the distance between point pairs for input layer data
            dis_latent {tensor} -- the distance between point pairs for latent layer data
            kNN_data {tensor} -- the mask to determine the neighborhood for input layer data
            kNN_latent {tensor} -- the mask to determine the neighborhood for latent layer data
        """

        # kNN_data = kNN_latent + kNN_data
        norml_data = torch.sqrt(torch.tensor(float(data.shape[1])))
        norml_latent = torch.sqrt(torch.tensor(float(latent.shape[1])))

        # Calculate Loss_iso
        D1_1 = (dis_data / norml_data)[kNN_data]
        D1_2 = (dis_latent / norml_latent)[kNN_data]
        Error1 = (D1_1 - D1_2) / 1
        loss_iso = torch.norm(Error1) / torch.sum(kNN_data)

        # Calculate Loss_push-away
        D2_1 = (dis_latent / norml_latent)[kNN_data is False]

        Error2 = (0 - torch.log(1 + D2_1)) / 1
        loss_push_away = torch.norm(Error2[Error2 > -1 * regular_B]) / max(
            torch.sum(kNN_data is False), 1
        )
        if epoch > chang_start:
            rate = max(
                rate_push
                - (epoch - chang_start) / (chang_end - chang_start) * rate_push,
                0,
            )
        else:
            rate = rate_push

        loss_push_away = -1.0 * rate * loss_push_away

        return loss_iso, loss_push_away
