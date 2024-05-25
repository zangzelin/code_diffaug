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


from dataloader.data_sourse import DigitsDataset
from dataloader.data_graph import *
from dataloader.data_insemb import *
from dataloader.data_XU import *
from dataloader.data_face import *
# from dataloader.data_DVIT import *


class Cifar10Dataset(DigitsDataset):
    def __init__(self, data_name="Cifar10", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        D = datasets.CIFAR10(root=datapath, train=True,
                           download=True, transform=None)

        # import pdb; pdb.set_trace()
        data = tensor(D.data)
        label = tensor(D.targets).reshape((-1))

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = False

class MnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D = datasets.MNIST(root=data_path, train=True, download=True, transform=None)

        data = (np.array(D.data[:60000]).astype(np.float32) / 255).reshape((60000, -1))
        label = np.array(D.targets[:60000]).reshape((-1))
        return data, label


class MnistALLDataset(DigitsDataset):
    def __init__(self, data_name="Mnist", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        D = datasets.MNIST(root=datapath, train=True,
                           download=True, transform=None)
        data_train = (D.data[:60000] / 255).float().reshape((60000, -1))
        label_train = (D.targets[:60000]).reshape((-1))
        D = datasets.MNIST(root=datapath, train=True,
                           download=True, transform=None)
        data_test = (D.data[:10000] / 255).float().reshape((10000, -1))
        label_test = (D.targets[:10000]).reshape((-1))
        
        fea_name = []
        for i in range(28):
            for j in range(28):
                fea_name.append('{}_{}'.format(i, j))

        self.feature_name = np.array(fea_name)
        self.def_fea_aim = 64
        # self.train_val_split(data, label, train)
        if train is True:
            self.data = data_train
            self.label = label_train
        else:
            self.data = data_test
            self.label = label_test
        self.graphwithpca = False


class Mnist10000Dataset(DigitsDataset):
    def __init__(self, data_name="Mnist", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        D = datasets.MNIST(root=datapath, train=True,
                           download=True, transform=None)

        data = (D.data[:10000] / 255).float().reshape((10000, -1))
        label = (D.targets[:10000]).reshape((-1))

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = False


class Mnist3000Dataset(DigitsDataset):
    def __init__(self, data_name="Mnist", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        D = datasets.MNIST(
            root=datapath, train=True,
            download=True, transform=None)

        data = (D.data[:3000] / 255).float().reshape((3000, -1))
        label = (D.targets[:3000]).reshape((-1))

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = False

class FMnistDataset(DigitsDataset):
    def __init__(self, data_name="FMnist", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        D = datasets.FashionMNIST(
            root=datapath, train=True, download=True, transform=None
        )

        data = (D.data[:20000] / 255).float().reshape((20000, -1))
        label = (D.targets[:20000]).reshape((-1))

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = False


class KMnistDataset(DigitsDataset):
    def __init__(self, data_name="KMnist", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        D = datasets.KMNIST(root=datapath, train=True,
                            download=True, transform=None)

        data = (D.data[:20000] / 255).float().reshape((20000, -1))
        label = (D.targets[:20000]).reshape((-1))

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = False


class EMnistDataset(DigitsDataset):
    def __init__(self, data_name="EMnist", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        D = datasets.EMNIST(root=datapath, train=True, split="byclass",
                            download=True, transform=None)

        data = (D.data[:20000] / 255).float().reshape((20000, -1))
        label = (D.targets[:20000]).reshape((-1))

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = False


class Coil20Dataset(DigitsDataset):
    def __init__(self, data_name="Coil20", train=True, datapath="/root/data"):
        # digit = load_digits()
        path = datapath + "/coil-20-proc"
        fig_path = os.listdir(path)
        fig_path.sort()

        self.data_name = data_name
        label = []
        data = np.zeros((1440, 128, 128))
        for i in range(1440):
            img = Image.open(path + "/" + fig_path[i])
            I_array = np.array(img)
            data[i] = I_array
            label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

        data = data.reshape((data.shape[0], -1)) / 255

        data = tensor(data).float()
        label = tensor(label).long()

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = False


class EMnistBCDataset(DigitsDataset):
    def __init__(self, data_name="EMnistBC", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        D = datasets.EMNIST(
            root=datapath,
            train=True,
            download=True,
            transform=None,
            split="byclass",
        )

        data = (D.data[:60000] / 255).float().reshape((60000, -1))
        label = (D.targets[:60000]).reshape((-1))

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = False


class ColonDataset(DigitsDataset):
    def __init__(self, data_name="Colon", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        sadata = sc.read(datapath + "/colonn.h5ad")

        data = tensor(sadata.X)
        label = tensor(np.array([int(i) for i in list(sadata.obs.clusterstr)]))

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = False


class arceneDataset(DigitsDataset):
    def __init__(self, data_name="arcene", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        # sadata = sc.read(datapath + "/colonn.h5ad")
        mat = scipy.io.loadmat(datapath+'/feature_select/arcene.mat')
        data = tensor(np.array(mat['X']).astype(np.float32)).float()
        label = tensor(np.array(mat['Y'])).long()

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = False

class Gast10k1457Dataset(DigitsDataset):
    def __init__(self, data_name="Gast10k1457", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        # sadata = sc.read(datapath + "/colonn.h5ad")
        sadata = sc.read(datapath+"/gast10kwithcelltype.h5ad")
        sadata_pca = np.array(sadata.X)
        # mat = scipy.io.loadmat(datapath+'/feature_select/arcene.mat')

        # sadata_pca = PCA(n_components=200).fit_transform(sadata_pca)

        data = torch.tensor(sadata_pca).float()

        label_train_str = list(sadata.obs['celltype'])
        label_train_str_set = list(set(label_train_str))
        label_train = torch.tensor(
            [label_train_str_set.index(i) for i in label_train_str])
        label = torch.tensor(label_train)

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True


class MCAD9119Dataset(DigitsDataset):
    def __init__(self, data_name="MCAD9119", train=True, datapath="~/data"):
        # digit = load_digits()

        self.data_name = data_name
        # sadata = sc.read(datapath + "/colonn.h5ad")
        data = np.load(datapath+'/mca_data/mca_data_dim_34947.npy')
        data = tensor(data[:, data.max(axis=0) > 4]).float()
        label = np.load(datapath+'/mca_data/mca_label_dim_34947.npy')

        # Count the number of each category in the label
        # label_count = Counter(label)
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
        # import pdb; pdb.set_trace()

        label = tensor(label)

        # import pdb; pdb.set_trace()
        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True


class HCL60K3037DDataset(DigitsDataset):
    def __init__(self, data_name="HCL60K3037D", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        # sadata = sc.read(datapath + "/colonn.h5ad")
        sadata = sc.read(datapath+"/HCL60kafter-elis-all.h5ad")
        data = tensor(sadata.X).float()
        # data = tensor(data[:, data.max(axis=0) > 4]).float()
        label = tensor(np.array([int(i) for i in list(sadata.obs.louvain)]))
        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True


class PBMCDataset(DigitsDataset):
    def __init__(self, data_name="PBMC", train=True, datapath="~/data"):
        self.data_name = data_name
        adata = sc.read(datapath+"/PBMC3k_HVG_regscale.h5ad")
        data = tensor(adata.obsm['X_pca'])
        label_train_str = list(adata.obs['celltype'])
        label_train_str_set = list(set(label_train_str))
        label = tensor(
            np.array([label_train_str_set.index(i) for i in label_train_str]))
        
        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True


class MiceProteinDataset(DigitsDataset):
    def __init__(self, data_name="MiceProtein", train=True, datapath="~/data"):
        self.data_name = data_name
        datameta = pd.read_csv(datapath+"/feature_select/Data_Cortex_Nuclear.csv")
        datameta = datameta.fillna(datameta.median())

        label = np.array(datameta['class'])
        data = np.array(datameta.drop(['MouseID', 'Genotype','Treatment','Behavior','class'], axis = 1))
        data = (data-data.min())/(data.max()-data.min())
        label_train_str_set = list(set(label))
        label_train = torch.tensor([label_train_str_set.index(i) for i in label])        
        
        data = tensor(data).float()
        label = tensor(label_train)

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True

# class Coil20Dataset(DigitsDataset):
#     def __init__(self, data_name="Coil20", train=True, datapath="~/data"):
#         self.data_name = data_name
        
#         path = datapath+"/coil-20-proc"
#         fig_path = os.listdir(path)
#         fig_path.sort()

#         label = []
#         data = np.zeros((1440, 128, 128))
#         for i in range(1440):
#             I = Image.open(path + "/" + fig_path[i])
#             I_array = np.array(I)
#             data[i] = I_array
#             label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

#         data = tensor(data).float().reshape(data.shape[0], -1)
#         label = tensor(label)

#         self.def_fea_aim = 64
#         self.train_val_split(data, label, train)
#         self.graphwithpca = True


class pixraw10PDataset(DigitsDataset):
    def __init__(self, data_name="pixraw10P", train=True, datapath="~/data"):
        self.data_name = data_name
        
        mat = scipy.io.loadmat(datapath+'/feature_select/pixraw10P.mat')

        data = np.array(mat['X']).astype(np.float32)
        data = (data-data.min())/(data.max()-data.min())
        label = np.array(mat['Y']).astype(np.int32)

        data = tensor(data).float().reshape(data.shape[0], -1)
        label = tensor(label)

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True

class ProstategeDataset(DigitsDataset):
    def __init__(self, data_name="Prostatege", train=True, datapath="~/data"):
        self.data_name = data_name
        
        mat = scipy.io.loadmat(datapath+'/feature_select/Prostate-GE.mat')

        data = np.array(mat['X']).astype(np.float32)
        data = (data-data.min())/(data.max()-data.min())
        label = np.array(mat['Y']).astype(np.int32)

        data = tensor(data).float().reshape(data.shape[0], -1)
        label = tensor(label)

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True


class ActivityDataset(DigitsDataset):
    def __init__(self, data_name="Activity", train=True, datapath="~/data"):
        self.data_name = data_name
        
        train_data = pd.read_csv(datapath+'/feature_select/Activity_train.csv')
        test_data = pd.read_csv(datapath+'/feature_select/Activity_test.csv')
        all_data = pd.concat([train_data, test_data])
        data = all_data.drop(['subject', 'Activity'], axis=1).to_numpy()
        label_str = all_data['Activity'].tolist()
        label_str_set = list(set(label_str))
        label = np.array([label_str_set.index(i) for i in label_str])
        data = (data-data.min())/(data.max()-data.min())

        data = tensor(data).float().reshape(data.shape[0], -1)
        label = tensor(label)

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True


class Coil100Dataset(DigitsDataset):
    def __init__(self, data_name="Coil100", train=True, datapath="~/data"):
        self.data_name = data_name
        
        path = datapath+"/coil-100"
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
        
        data = tensor(data).float().reshape(data.shape[0], -1)/ 255
        label = tensor(label)
        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True



class PBMCD2638Dataset(DigitsDataset):
    def __init__(self, data_name="PBMCD2638", train=True, datapath="~/data"):
        self.data_name = data_name
        
        adata = sc.read(datapath+"/PBMC3k_HVG_regscale.h5ad")
        data = np.array(adata.X)
        label_train_str = list(adata.obs['celltype'])
        label_train_str_set = list(set(label_train_str))
        label = np.array([label_train_str_set.index(i) for i in label_train_str])
        
        data = np.delete(data, 2053, axis=0)
        label = np.delete(label, 2053, axis=0)

        data = tensor(data).float().reshape(data.shape[0], -1)
        label = tensor(label)

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True


class OTUDataset(DigitsDataset):
    def __init__(self, data_name="OTU", train=True, datapath="~/data"):
        self.data_name = data_name
        
        adata = pd.read_csv(datapath+"/PeiData/OTU.txt", sep = '\t' ).drop('eid', axis=1)
        alab = pd.read_csv(datapath+"/PeiData/OTU_lable.txt", sep = '\t').drop('eid', axis=1)
    
        data = adata.to_numpy()
        label = alab.to_numpy()
        data = tensor(data).float().reshape(data.shape[0], -1)
        label = tensor(label)

        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True


class InsEmb_ExtyaleBDataset(DigitsDataset):
    def __init__(self, data_name="ExtyaleB", train=True, datapath="~/data/Datasets_dr/Datasets_dr"):
        self.data_name = data_name
        data = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/ExtyaleB/ExtyaleB.csv').to_numpy())).float()
        label = tensor(
            StandardScaler().fit_transform(pd.read_csv(
                datapath+'/Datasets_dr/ExtyaleB/ExtyaleB-label.csv'
                ).to_numpy())).float()+1
        self.train_val_split(data, label, train)
        self.graphwithpca = False

class InsEmb_mnist64Dataset(DigitsDataset):
    def __init__(self, data_name="mnist64", train=True, datapath="~/Datasets_dr/Datasets_dr/Datasets_dr"):
        self.data_name = data_name
        data = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/mnist64/mnist64.csv').to_numpy())).float()
        label = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/mnist64/mnist64-label.csv').to_numpy())).float()
        self.train_val_split(data, label, train)
        self.graphwithpca = False

class InsEmb_bostonDataset(DigitsDataset):
    def __init__(self, data_name="boston", train=True, datapath="~/data/Datasets_dr/Datasets_dr"):
        self.data_name = data_name
        data = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/boston/boston.csv').to_numpy())).float()
        label = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/boston/boston-label.csv').to_numpy())).float()
        self.train_val_split(data, label, train)
        self.graphwithpca = False

class InsEmb_dermatologyDataset(DigitsDataset):
    def __init__(self, data_name="dermatology", train=True, datapath="~/data/Datasets_dr/Datasets_dr"):
        self.data_name = data_name
        data = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/dermatology/dermatology.csv').to_numpy())).float()
        label = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/dermatology/dermatology-label.csv').to_numpy())).float()
        self.train_val_split(data, label, train)
        self.graphwithpca = False
    
class InsEmb_ecoliDataset(DigitsDataset):
    def __init__(self, data_name="ecoli", train=True, datapath="~/data/Datasets_dr/Datasets_dr"):
        self.data_name = data_name
        data = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/ecoli/ecoli.csv').to_numpy())).float()
        label = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/ecoli/ecoli-label.csv').to_numpy())).float()
        self.train_val_split(data, label, train)
        self.graphwithpca = False

class InsEmb_oliveDataset(DigitsDataset):
    def __init__(self, data_name="olive", train=True, datapath="~/data/Datasets_dr/Datasets_dr"):
        self.data_name = data_name
        data = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/olive/olive.csv').to_numpy())).float()
        label = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/olive/olive-label.csv').to_numpy())).float()
        self.train_val_split(data, label, train)
        self.graphwithpca = False

class InsEmb_weatherDataset(DigitsDataset):
    def __init__(self, data_name="weather", train=True, datapath="~/data/Datasets_dr/Datasets_dr"):
        self.data_name = data_name
        data = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/weather/weather.csv').to_numpy())).float()
        label = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/weather/weather-label.csv').to_numpy())).float()
        self.train_val_split(data, label, train)
        self.graphwithpca = False

class InsEmb_world12dDataset(DigitsDataset):
    def __init__(self, data_name="world12d", train=True, datapath="~/data/Datasets_dr/Datasets_dr"):
        self.data_name = data_name
        data = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/world12d/world12d.csv').to_numpy())).float()
        label = tensor(StandardScaler().fit_transform(pd.read_csv(datapath+'/Datasets_dr/world12d/world12d-label.csv').to_numpy())).float()
        self.train_val_split(data, label, train)
        self.graphwithpca = False

class YONGJIE_UCDataset(DigitsDataset):
    def __init__(self, data_name="YONGJIE_UC", train=True, datapath="~/data"):

        self.data_name = data_name
        data = np.array(scipy.io.mmread(datapath+"/UC/uc_epi.mtx").todense()).T
        label = pd.read_csv(datapath+"/UC/uc_epi_celltype.tsv", header=None).to_numpy()

        label_train_str_set = list(set(label[:,0].tolist()))
        label = np.array([label_train_str_set.index(i) for i in label[:,0]])
        data = tensor(data)
        label = tensor(label)
        
        self.def_fea_aim = 64
        self.train_val_split(data, label, train)
        self.graphwithpca = True