from torch.utils import data
from sklearn.datasets import load_digits
from torch import tensor
import torchvision.datasets as datasets
from pynndescent import NNDescent
import os
import joblib
import torch
import numpy as np
from PIL import Image
import scanpy as sc
import scipy, sys
from sklearn.decomposition import PCA
import pandas as pd
import scipy.sparse as sp

import pickle as pkl

from dataloader.data_sourse import DigitsDataset


class GRAPH_CORADataset(DigitsDataset):

    def _ParseIndexFile(self, filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def _FeatureNormalize(self, mx):
        import scipy.sparse as sp
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        rowsum = (rowsum == 0) * 1 + rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


    def __init__(self, data_name="GRAPH_CORA", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name

        path_data = datapath+"/graphdata"
        names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
        objects = []
        data_name = 'cora'
        for i in range(len(names)):
            with open(
                path_data + "/ind.{}.{}".format(data_name, names[i]), "rb"
            ) as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding="latin1"))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = self._ParseIndexFile(
            path_data + "/ind.{}.test.index".format(data_name))

        test_idx_range = np.sort(test_idx_reorder)
        features = sp.vstack((allx, tx)).tolil()

        graph_eq = np.zeros(shape=(features.shape[0], 100))
        for i in range(len(graph)):
            if len(graph[i])>0:
                repeat_nodeindex = graph[i]*100
            else:
                repeat_nodeindex = [i]*100
            # print(i)
            # print(repeat_nodeindex)
            graph_eq[i,:] = np.concatenate(
                [
                    np.array([i]),
                    np.array(repeat_nodeindex)
                ])[:100]

        graph_eq = graph_eq.astype(np.int32)
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        features = self._FeatureNormalize(features)
        data_train = torch.FloatTensor(np.array(features.todense())).float()
        labels = torch.LongTensor(labels)
        label_train = torch.max(labels, dim=1)[1]

        data = tensor(data_train).float()
        label = tensor(label_train)

        self.graph_eq = graph_eq
        self.def_fea_aim = 64
        self.train_val_split(data, label, train, split_int=5)
        self.graphwithpca = False