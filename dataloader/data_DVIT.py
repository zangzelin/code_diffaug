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
import scipy
from sklearn.decomposition import PCA
import sklearn
import pandas as pd


from dataloader.data_sourse import DigitsDataset
from dataloader.data_graph import *
from dataloader.data_insemb import *
from dataloader.data_XU import *
from dataloader.data_face import *


class MnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D = datasets.MNIST(root=data_path, train=True, download=True, transform=None)

        data = (np.array(D.data[:60000]).astype(np.float32) / 255).reshape((60000, -1))
        label = np.array(D.targets[:60000]).reshape((-1))
        return data, label


class Mnistt10Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D = datasets.MNIST(root=data_path, train=True, download=True, transform=None)

        data = (np.array(D.data[:60000]).astype(np.float32) / 255).reshape((60000, -1))
        label = np.array(D.targets[:60000]).reshape((-1))

        data = np.concatenate(
            [data, data, data, data, data, data, data, data, data, data], axis=0
        )
        label = np.concatenate(
            [label, label, label, label, label, label, label, label, label, label],
            axis=0,
        )
        return data, label


class ActivityDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_data = pd.read_csv(data_path + "/feature_select/Activity_train.csv")
        test_data = pd.read_csv(data_path + "/feature_select/Activity_test.csv")
        all_data = pd.concat([train_data, test_data])
        data = all_data.drop(["subject", "Activity"], axis=1).to_numpy()
        label_str = all_data["Activity"].tolist()
        label_str_set = list(set(label_str))
        label = np.array([label_str_set.index(i) for i in label_str])
        data = (data - data.min()) / (data.max() - data.min())

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class Must_1_Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/V1_Adult_Mouse_Brain/emb.npy")
        label = np.load(data_path + "/V1_Adult_Mouse_Brain/pred_main.npy")
        # import pdb; pdb.set_trace()

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class SAMUSIKDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        train_data = pd.read_csv(data_path + "/feature_select/Activity_train.csv")
        test_data = pd.read_csv(data_path + "/feature_select/Activity_test.csv")
        all_data = pd.concat([train_data, test_data])
        data = all_data.drop(["subject", "Activity"], axis=1).to_numpy()
        label_str = all_data["Activity"].tolist()
        label_str_set = list(set(label_str))
        label = np.array([label_str_set.index(i) for i in label_str])
        data = (data - data.min()) / (data.max() - data.min())

        data = np.array(data).astype(np.float32).reshape(data.shape[0], -1)
        label = np.array(label)
        print("data.shape", data.shape)
        return data, label


class KMnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D = datasets.KMNIST(root=data_path, train=True, download=True, transform=None)

        data = (D.data[:60000] / 255).float().reshape((60000, -1))
        label = (D.targets[:60000]).reshape((-1))
        return data, label


class EMnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D = datasets.EMNIST(
            root=data_path, train=True, split="byclass", download=True, transform=None
        )

        data = (D.data[:280000] / 255).float().reshape((280000, -1))
        label = (D.targets[:280000]).reshape((-1))
        return data, label


class Coil20Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        path = data_path + "/coil-20-proc"
        fig_path = os.listdir(path)
        fig_path.sort()

        label = []
        data = np.zeros((1440, 128, 128))
        for i in range(1440):
            img = Image.open(path + "/" + fig_path[i])
            I_array = np.array(img)
            data[i] = I_array
            label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

        data = data.reshape((data.shape[0], -1)) / 255
        data = data.astype(np.float32)
        label = np.array(label)
        return data, label


class Coil100Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        path = data_path + "/coil-100"
        fig_path = os.listdir(path)

        label = []
        data = np.zeros((100 * 72, 128, 128, 3))
        for i, path_i in enumerate(fig_path):
            # print(i)
            if "obj" in path_i:
                I = Image.open(path + "/" + path_i)
                I_array = np.array(I.resize((128, 128)))
                data[i] = I_array
                label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

        data = (data.astype(np.float32) / 255).reshape(data.shape[0], -1)
        label = np.array(label)
        return data, label


class Cifar10Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D = datasets.CIFAR10(root=data_path, train=True, download=True, transform=None)

        data = np.array(D.data).astype(np.uint8)
        label = np.array(D.targets).reshape((-1))
        return data, label


class Cifar100Dataset(DigitsDataset):
    def cifar100labeltrans(self, label):
        coarse_labels = np.array(
            [
                4,
                1,
                14,
                8,
                0,
                6,
                7,
                7,
                18,
                3,
                3,
                14,
                9,
                18,
                7,
                11,
                3,
                9,
                7,
                11,
                6,
                11,
                5,
                10,
                7,
                6,
                13,
                15,
                3,
                15,
                0,
                11,
                1,
                10,
                12,
                14,
                16,
                9,
                11,
                5,
                5,
                19,
                8,
                8,
                15,
                13,
                14,
                17,
                18,
                10,
                16,
                4,
                17,
                4,
                2,
                0,
                17,
                4,
                18,
                17,
                10,
                3,
                2,
                12,
                12,
                16,
                12,
                1,
                9,
                19,
                2,
                10,
                0,
                1,
                16,
                12,
                9,
                13,
                15,
                13,
                16,
                19,
                2,
                4,
                6,
                19,
                5,
                5,
                8,
                19,
                18,
                1,
                2,
                15,
                6,
                0,
                17,
                8,
                14,
                13,
            ]
        )
        return coarse_labels[label]

    def load_data(self, data_path, train=True):
        D = datasets.CIFAR100(root=data_path, train=True, download=True, transform=None)
        data = np.array(D.data).astype(np.uint8)
        label = np.array(D.targets).reshape((-1))
        label = self.cifar100labeltrans(label)
        return data, label


class HCL60KDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        sadata = sc.read(data_path + "/HCL60kafter-elis-all.h5ad")
        data = np.array(sadata.X).astype(np.float32)
        label = np.array(np.array([int(i) for i in list(sadata.obs.louvain)]))
        index_fea = (data<6.9341908).sum(axis=0)<59700
        data = data[:, index_fea]/10
        # import pdb; pdb.set_trace()

        # import MinMaxScaler
        # scaler = sklearn.preprocessing.MinMaxScaler()
        # data = scaler.fit_transform(data)

        mask_label = (np.zeros(label.shape) + 1).astype(np.bool_)
        for l in range(label.max() + 1):
            num_l = (label == l).sum()
            if num_l < 500:
                mask_label[label == l] = False

        data = data[mask_label]
        label = label[mask_label]

        return data, label


class HCL60KPCA500Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        sadata = sc.read(data_path + "/HCL60kafter-elis-all.h5ad")
        data = np.array(sadata.X).astype(np.float32)
        label = np.array(np.array([int(i) for i in list(sadata.obs.louvain)]))

        # import MinMaxScaler
        scaler = PCA(n_components=500)
        data = scaler.fit_transform(data)

        # import MinMaxScaler
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
        data = scaler.fit_transform(data)
        # import pdb; pdb.set_trace()

        mask_label = (np.zeros(label.shape) + 1).astype(np.bool_)
        for l in range(label.max() + 1):
            num_l = (label == l).sum()
            if num_l < 500:
                mask_label[label == l] = False

        data = data[mask_label]
        label = label[mask_label]

        return data, label


class MCAD9119Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/mca_data/mca_data_dim_34947.npy")
        data = (data[:, data.max(axis=0) > 4]).astype(np.float32)
        label = np.load(data_path + "/mca_data/mca_label_dim_34947.npy")

        label_count = {}
        for i in label:
            if i in label_count:
                label_count[i] += 1
            else:
                label_count[i] = 1
        # if the number of each category is less than 100, delete this category
        for i in label_count:
            if label_count[i] < 100:
                label[label == i] = -1
        # delete the data with label -1
        data = data[label != -1]
        label = label[label != -1]

        return data, label.astype(np.int32)

class MCAD1374D8Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/mca_data/mca_data_dim_34947.npy")
        select_index = (data!=0.0).sum(axis=0) > 3000
        data = (data[:, select_index]).astype(np.float32)
        label = np.load(data_path + "/mca_data/mca_label_dim_34947.npy")

        label_count = {}
        for i in label:
            if i in label_count:
                label_count[i] += 1
            else:
                label_count[i] = 1
        # if the number of each category is less than 100, delete this category
        for i in label_count:
            if label_count[i] < 100:
                label[label == i] = -1
        # delete the data with label -1
        data = data[label != -1]/8.0
        label = label[label != -1]
        
        return data, label.astype(np.int32)


class MCAD251D5Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        data = np.load(data_path + "/mca_data/mca_data_dim_34947.npy")
        select_index = (data!=0.0).sum(axis=0) > 10000
        # import pdb; pdb.set_trace()
        data = (data[:, select_index]).astype(np.float32)
        label = np.load(data_path + "/mca_data/mca_label_dim_34947.npy")
        
        label_count = {}
        for i in label:
            if i in label_count:
                label_count[i] += 1
            else:
                label_count[i] = 1
        # if the number of each category is less than 100, delete this category
        for i in label_count:
            if label_count[i] < 100:
                label[label == i] = -1
        # delete the data with label -1
        data = data[label != -1]/5.0
        label = label[label != -1]
        
        return data, label.astype(np.int32)


class Gast10k1457Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        sadata = sc.read(data_path + "/gast10kwithcelltype.h5ad")
        sadata_pca = np.array(sadata.X)
        data = np.array(sadata_pca).astype(np.float32)

        label_train_str = list(sadata.obs["celltype"])
        label_train_str_set = list(set(label_train_str))
        label_train = torch.tensor(
            [label_train_str_set.index(i) for i in label_train_str]
        )
        label = np.array(label_train).astype(np.int32)

        return data, label

class Gast10k1457D10Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        sadata = sc.read(data_path + "/gast10kwithcelltype.h5ad")
        sadata_pca = np.array(sadata.X)
        data = np.array(sadata_pca).astype(np.float32)/10

        label_train_str = list(sadata.obs["celltype"])
        label_train_str_set = list(set(label_train_str))
        label_train = torch.tensor(
            [label_train_str_set.index(i) for i in label_train_str]
        )
        label = np.array(label_train).astype(np.int32)

        return data, label

class Gast10k1457D8Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        sadata = sc.read(data_path + "/gast10kwithcelltype.h5ad")
        sadata_pca = np.array(sadata.X)
        data = np.array(sadata_pca).astype(np.float32)/8

        label_train_str = list(sadata.obs["celltype"])
        label_train_str_set = list(set(label_train_str))
        label_train = torch.tensor(
            [label_train_str_set.index(i) for i in label_train_str]
        )
        label = np.array(label_train).astype(np.int32)

        label_count = {}
        for i in label:
            if i in label_count:
                label_count[i] += 1
            else:
                label_count[i] = 1
        # if the number of each category is less than 100, delete this category
        for i in label_count:
            if label_count[i] < 50:
                label[label == i] = -1
        # delete the data with label -1
        data = data[label != -1]
        label = label[label != -1]

        return data, label

class Gast10k700Dataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        sadata = sc.read(data_path + "/gast10kwithcelltype.h5ad")
        sadata_pca = np.array(sadata.X)
        data = np.array(sadata_pca).astype(np.float32)

        label_train_str = list(sadata.obs["celltype"])
        label_train_str_set = list(set(label_train_str))
        label_train = torch.tensor(
            [label_train_str_set.index(i) for i in label_train_str]
        )
        label = np.array(label_train).astype(np.int32)
        return data, label