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
import pandas as pd

# import PoolRunner
import dataloader.cal_sigma as cal_sigma
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import random
from torchvision import transforms as transform_lib

# from kornia import image_to_tensor, tensor_to_image
# from kornia.augmentation import Normalize, RandomGaussianBlur, RandomSolarize, RandomGrayscale, RandomResizedCrop, RandomCrop, ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline
# from kornia import augmentation as kornia_aug
# from kornia.geometry import transform as kornia_tf
# from kornia import filters as kornia_filter


class DigitsDataset(data.Dataset):
    def __init__(
        self,
        data_name="Digits",
        train=True,
        data_path="/zangzelin/data",
        k=10,
        pca_dim=100,
        uselabel=False,
        n_cluster=25,
        n_f_per_cluster=3,
        l_token=50,
        data_sample=5000,
        uniform_param=1,
        seed=0,
        preprocess_bool=True,
        rrc_rate=0.8,
        trans_range=2,
    ):
        self.data_name = data_name
        self.train = train
        data, label = self.load_data(data_path=data_path)

        if "Cifar" in self.data_name:
            solarize_prob = 0.0
            blur_prob = 1.0
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            color_jitter = transform_lib.ColorJitter(0.4, 0.4, 0.2, 0.1)
            normalize = transform_lib.Normalize(mean=mean, std=std)
            crop_size = 32

            self.aug_trans = transform_lib.Compose(
                [
                    transform_lib.RandomResizedCrop(crop_size),
                    transform_lib.RandomHorizontalFlip(),
                    transform_lib.RandomApply([color_jitter], p=0.8),
                    transform_lib.RandomGrayscale(p=0.2),
                    # transform_lib.RandomApply([transform_lib.GaussianBlur(kernel_size=23)], p=blur_prob),
                    # transform_lib.RandomApply([Solarize()], p=solarize_prob),
                    # transform_lib.RandomSolarize(threshold=128, p=solarize_prob),
                    transform_lib.ToTensor(),
                    normalize,
                ]
            )

        np.random.seed(seed)
        if "Cifar" not in self.data_name:
            filename = f"save_near_index/token_index_dataname{self.data_name}n_cluster{n_cluster}n_f_per_cluster{n_f_per_cluster}l_token{l_token}_data_sample{data_sample}.pkl"
            if not os.path.exists(filename):
                token_index = self.tokenisation(
                    data,
                    n_cluster=n_cluster,
                    n_f_per_cluster=n_f_per_cluster,
                    l_token=l_token,
                    data_sample=data_sample,
                )
                joblib.dump(token_index, filename)
            else:
                token_index = joblib.load(filename)
            # print('token_index', token_index)
            self.token_index = token_index

        np.random.seed(seed)
        rand_index = np.random.permutation(data.shape[0])
        if train:
            rand_index = rand_index[: int(rrc_rate * data.shape[0])]
        else:
            rand_index = rand_index[int(rrc_rate * data.shape[0]) :]

        data = data[rand_index]
        label = label[rand_index]
        # import pdb; pdb.set_trace()
        self.data = np.array(data)
        self.label = np.array(label)
        self.uniform_param = uniform_param

        if "Cifar" not in self.data_name and preprocess_bool:
            neighbors_index, n_token_feature = self.Preprocessing(
                data=data.reshape(data.shape[0], -1),
                k=k,
                pca_dim=pca_dim,
                uselabel=uselabel,
                token_index=token_index,
            )
            self.neighbors_index = neighbors_index
            self.n_token_feature = n_token_feature
        # import pdb; pdb.set_trace()
        # trans_range = 6
        translate_range = [trans_range, trans_range]
        print("translate", translate_range)
        width = 28
        # self.transform_aug_mnist = transform_lib.RandomResizedCrop(28, scale=(rrc_rate, 1.0), ratio=(1.0,1.0))
        self.transform_aug_mnist = transform_lib.RandomAffine(
            degrees=0,
            translate=(translate_range[0] / width, translate_range[1] / width),
        )
        # import pdb; pdb.set_trace()

    def load_data(self, data_path):
        digit = load_digits()
        data = np.array(digit.data).astype(np.float32)
        label = np.array(digit.target)
        return data, label

    def Preprocessing(self, data, k, pca_dim, uselabel, token_index):
        self.graphwithpca = False

        neighbors_index = self.cal_near_index(
            k=k,
            uselabel=uselabel,
            pca_dim=pca_dim,
        )

        n_token_feature = token_index.shape[0]
        # lat_list = []
        # for i in range(token_index.shape[0]):
        #     lat_list.append(data[:, None, token_index[i]])
        # data_input = np.concatenate(lat_list, axis=1)

        return neighbors_index, n_token_feature

    def cal_near_index(self, k=10, device="cuda", uselabel=False, pca_dim=100):
        filename = "save_near_index/data_name{}K{}uselabel{}pcadim{}.pkl".format(
            self.data_name, k, uselabel, pca_dim
        )
        os.makedirs("save_near_index", exist_ok=True)
        if not os.path.exists(filename):
            X_rshaped = self.data.reshape((self.data.shape[0], -1))
            if pca_dim < X_rshaped.shape[1]:
                X_rshaped = PCA(n_components=pca_dim).fit_transform(X_rshaped)
            if not uselabel:
                index = NNDescent(X_rshaped, n_jobs=-1)
                neighbors_index, neighbors_dist = index.query(X_rshaped, k=k + 1)
                neighbors_index = neighbors_index[:, 1:]
            else:
                dis = pairwise_distances(X_rshaped)
                M = np.repeat(self.label.reshape(1, -1), X_rshaped.shape[0], axis=0)
                dis[(M - M.T) != 0] = dis.max() + 1
                neighbors_index = dis.argsort(axis=1)[:, 1 : k + 1]
            joblib.dump(value=neighbors_index, filename=filename)

            print("save data to ", filename)
        else:
            print("load data from ", filename)
            neighbors_index = joblib.load(filename)
        return neighbors_index

    def tokenisation(
        self, data, n_cluster=25, n_f_per_cluster=3, l_token=256, data_sample=5000
    ):
        if data.shape[0] > data_sample:
            index_rand_downsample = np.arange(data.shape[0])
            np.random.shuffle(index_rand_downsample)
            index_rand_downsample = index_rand_downsample[:data_sample]
            data = data[index_rand_downsample]
        if data.shape[1] < l_token:
            l_token = data.shape[1]
        if n_cluster >= data.shape[0]:
            n_cluster = data.shape[1] - 1

        kmeans = KMeans(
            n_clusters=n_cluster,
            random_state=0,
        ).fit(data.T)
        dic_clu = {}
        for i in kmeans.labels_:  # 把784组特征聚为25组高纬语义特征
            dic_clu[i] = np.where(kmeans.labels_ == i)[0]

        index_selected_fea_np = (
            np.zeros((n_cluster, data.shape[1])) - 1
        )  # 25*50 50个特征都属于25个类别中的哪一类
        for i in range(n_cluster):
            index_selected_fea_np[i, : len(dic_clu[i])] = np.array(dic_clu[i])
        index_selected_fea = list(
            index_selected_fea_np[:, :n_f_per_cluster].reshape((-1)).astype(np.int32)
        )  # 为什么取前四个
        index_selected_fea = list(set(index_selected_fea))
        print("len(index_selected_fea)", len(index_selected_fea))

        data = data + np.abs(
            np.random.randn(data.shape[0], data.shape[1]) / 1000
        )  # 为什么？
        c_data = np.corrcoef(data.T)
        sorted_indices = np.argsort(-c_data)
        selected_indices = sorted_indices[index_selected_fea]
        t_index = selected_indices[:, :l_token]
        return t_index

    # def to_device(self, device):
    #     self.labelstr = [[str(int(i)) for i in self.label]]
    #     self.data = self.data.to(device)
    #     self.label = self.label.to(device)

    def __getitem__(self, index):
        if self.train:
            if "Cifar" in self.data_name:
                # data = Image.fromarray(self.data[index])
                # data_input_item = self.aug_trans(data).reshape((-1))
                # data_input_aug = self.aug_trans(data).reshape((-1))
                # data = Image.fromarray(self.data[index])
                data_input_item = self.data[index].astype(np.float32)
                data_input_aug = self.data[index].astype(np.float32)

                label = self.label[index]
                self.token_index = np.arange(0, data_input_item.shape[0])

            else:
                data_input_item = self.data[index].astype(np.float32)

                neighbor_index_list = self.neighbors_index[index]
                randi = np.random.choice(neighbor_index_list.shape[0], 1)[0]
                selected_index = neighbor_index_list[randi]
                if "Mnist" in self.data_name:
                    pil_img = Image.fromarray(
                        self.data[selected_index].reshape(28, 28) * 255
                    ).convert("L")
                    data_new = self.transform_aug_mnist(pil_img)

                    # save the data_new
                    # data_old = np.array(data_input_item)
                    # data_old = data_old.reshape(28, 28)
                    # data_new = np.array(data_new)
                    # data_new = data_new.reshape(28, 28)
                    # import matplotlib.pyplot as plt
                    # plt.subplot(1,2,1)
                    # plt.imshow(data_old)
                    # plt.subplot(1,2,2)
                    # plt.imshow(data_new)
                    # plt.savefig(f'da_aug{index}.png')

                    data_input_aug_fro = (
                        np.array(data_new).astype(np.float32).reshape(784)
                    ) / 255
                    # import pdb; pdb.set_trace()
                else:
                    data_input_aug_fro = self.data[selected_index]
                    # import pdb; pdb.set_trace()
                alpha = np.random.uniform(0, self.uniform_param, 1).astype(np.float32)
                data_input_aug = data_input_item * alpha + data_input_aug_fro * (
                    1 - alpha
                )
                label = self.label[index]
                # print('data_input_aug', data_input_aug.shape)
            return (
                (data_input_item, data_input_aug, data_input_item),
                label,
                index,
                self.token_index,
            )
        else:
            if "Cifar" in self.data_name:
                data_input_item = self.data[index].astype(np.float32)
            else:
                data_input_item = self.data[index]
            label = self.label[index]
            return data_input_item, label, index, self.token_index

    def __len__(self):
        return self.data.shape[0]

    def get_dim(
        self,
    ):
        return self.data[0].shape

    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b
        return out

    def _CalRho(self, dist):
        dist_copy = np.copy(dist)
        row, col = np.diag_indices_from(dist_copy)
        dist_copy[row, col] = 1e16
        rho = np.min(dist_copy, axis=1)
        return rho

    def get_sigma_rho(self, X, perplexity, v_input, K=500):
        print("use kNN mehtod to find the sigma")

        X_rshaped = X.reshape((X.shape[0], -1))

        if X_rshaped.shape[1] > 100:
            X_rshaped = PCA(n_components=50).fit_transform(X_rshaped)
            print("--------------->PCA", X_rshaped.shape)

        # index = NNDescent(X_rshaped, n_jobs=-1,)
        # neighbors_index, neighbors_dist = index.query(X_rshaped, k=K )
        # neighbors_dist = np.power(neighbors_dist, 2)
        # rho = neighbors_dist[:, 1]

        dist = np.power(
            pairwise_distances(
                X.reshape((X.shape[0], -1)),
                n_jobs=-1,
            ),
            2,
        )
        rho = self._CalRho(dist)

        r = cal_sigma.PoolRunner(
            number_point=X.shape[0],
            perplexity=perplexity,
            dist=dist,
            rho=rho,
            gamma=self._CalGamma(v_input),
            v=v_input,
            pow=2,
        )
        sigma = np.array(r.Getout())

        std_dis = np.std(rho) / np.sqrt(X.shape[1])
        print("sigma", sigma)
        print("sigma max", np.max(sigma))
        # if std_dis < 0.20 or self.same_sigma is True:
        #     # sigma[:] = sigma.mean() * 5
        #     sigma[:] = sigma.mean()
        # print('sigma', sigma)
        return rho, sigma
