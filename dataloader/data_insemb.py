from torch.utils import data
from sklearn.datasets import load_digits
from torch import tensor
import torchvision.datasets as datasets
from pynndescent import NNDescent
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from dataloader.data_sourse import DigitsDataset

class InsEmb_Car2Dataset(DigitsDataset):
    def __init__(self, data_name="InsEmb_Car2", train=True, datapath="~/data"):
        self.data_name = data_name
        data_raw = pd.read_csv(datapath+'/feature_emb/car2.csv').drop(['Name'], axis=1).to_numpy()
        data = tensor(StandardScaler().fit_transform(data_raw)).float()
        label = tensor([0]*data.shape[0])
        
        self.def_fea_aim = 50
        self.index_s = list(range(data.shape[1]))
        self.train_val_split(data, label, train, split_int=5)
        self.graphwithpca = False

class InsEmb_UnivDataset(DigitsDataset):
    def __init__(self, data_name="InsEmb_Univ", train=True, datapath="~/data"):
        self.data_name = data_name
        data_raw = pd.read_csv(datapath+'/feature_emb/univ.csv').drop(['Name'], axis=1).to_numpy()
        data = tensor(StandardScaler().fit_transform(data_raw)).float()
        label = tensor([0]*data.shape[0])
        
        self.def_fea_aim = 50
        self.index_s = list(range(data.shape[1]))
        self.train_val_split(data, label, train, split_int=5)
        self.graphwithpca = False


def Get_protein_index(protein_dict, protein_list):
    index_list = []
    for item in protein_list:
        index_list.append(protein_dict.index(item))
    return index_list
def Get_selected_protein_matrix(proteins, selected_protein_index):
    return proteins[:, selected_protein_index]

def Fill_nan(data, fill_value):
    data[np.isnan(data)] = fill_value
    return data

class InsEmb_TPD_867Dataset(DigitsDataset):
    
    def Splitdata(self, patient_all, label_all, protein_all, set_all, splitstr):

        choose_bool = [True if set_item == splitstr else False for set_item in set_all]
        # print(choose_bool)
        
        patient_new = []
        label_new = []

        for i in range(len(set_all)):
            if choose_bool[i] == True:
                patient_new.append(patient_all[i])
                label_new.append(label_all[i])

        protein_new = protein_all[choose_bool,:]
        return patient_new, label_new, protein_new

    def __init__(self, data_name="InsEmb_TPD_867", train=True, datapath="~/data"):
        self.data_name = data_name

        data_frame = pd.read_csv(datapath+'/alldata.csv')
        patient_all = data_frame['Unnamed: 0'].tolist()
        label_all_str = data_frame['label'].tolist()
        label_all = [1.0 if item == 'B' else 0.0 for item in label_all_str]
        set_all = data_frame['Sets'].tolist()
        protein_all = data_frame.drop(['Unnamed: 0', 'label', 'Sets'], axis=1).to_numpy()
        protein_name = data_frame.columns.to_list()
        protein_name.remove('label')
        protein_name.remove('Unnamed: 0')
        protein_name.remove('Sets')
        patient_dis, label_dis, protein_dis = self.Splitdata(patient_all, label_all, protein_all, set_all, 'Discovery')
        patient_ret, label_ret, protein_ret = self.Splitdata(patient_all, label_all, protein_all, set_all, 'Retrospective Test')
        patient_dis = np.concatenate([patient_dis, patient_ret])
        label_dis = np.concatenate([label_dis, label_ret])
        protein_dis = np.concatenate([protein_dis, protein_ret])
        protein_list = [
            "P02765","P04083","O00339","P58546","O75347","P04216","P02751",
            "P83731","P00568","P78527","P04792","P57737","P42224","P27797",
            "Q9HAT2","P30086","O14964","P10909","P17931"
            ]
        index_list = Get_protein_index(protein_name, protein_list)
        data_dis = Get_selected_protein_matrix(protein_dis, index_list)
        data_dis = Fill_nan(data_dis, 12)
        data = tensor(data_dis).float()
        label = tensor(label_dis)
        
        self.def_fea_aim = 50
        self.train_val_split(data, label, train, split_int=5)
        self.graphwithpca = False


class InsEmb_TPD_579Dataset(DigitsDataset):
    
    def Splitdata(self, patient_all, label_all, protein_all, set_all, splitstr):

        choose_bool = [True if set_item == splitstr else False for set_item in set_all]
        # print(choose_bool)
        
        patient_new = []
        label_new = []

        for i in range(len(set_all)):
            if choose_bool[i] == True:
                patient_new.append(patient_all[i])
                label_new.append(label_all[i])

        protein_new = protein_all[choose_bool,:]
        return patient_new, label_new, protein_new

    def __init__(self, data_name="InsEmb_TPD_579", train=True, datapath="~/data"):
        self.data_name = data_name

        data_frame = pd.read_csv(datapath+'/alldata.csv')
        patient_all = data_frame['Unnamed: 0'].tolist()
        label_all_str = data_frame['label'].tolist()
        label_all = [1.0 if item == 'B' else 0.0 for item in label_all_str]
        set_all = data_frame['Sets'].tolist()
        protein_all = data_frame.drop(['Unnamed: 0', 'label', 'Sets'], axis=1).to_numpy()
        protein_name = data_frame.columns.to_list()
        protein_name.remove('label')
        protein_name.remove('Unnamed: 0')
        protein_name.remove('Sets')
        patient_dis, label_dis, protein_dis = self.Splitdata(patient_all, label_all, protein_all, set_all, 'Discovery')
        # patient_ret, label_ret, protein_ret = self.Splitdata(patient_all, label_all, protein_all, set_all, 'Retrospective Test')
        patient_dis = np.concatenate([patient_dis])
        label_dis = np.concatenate([label_dis])
        protein_dis = np.concatenate([protein_dis])
        protein_list = [
            "P02765","P04083","O00339","P58546","O75347","P04216","P02751",
            "P83731","P00568","P78527","P04792","P57737","P42224","P27797",
            "Q9HAT2","P30086","O14964","P10909","P17931"
            ]
        index_list = Get_protein_index(protein_name, protein_list)
        data_dis = Get_selected_protein_matrix(protein_dis, index_list)
        data_dis = Fill_nan(data_dis, 12)
        data = tensor(data_dis).float()
        label = tensor(label_dis)
        
        self.def_fea_aim = 50
        self.index_s = list(range(data.shape[1]))
        self.train_val_split(data, label, train, split_int=4)
        self.graphwithpca = False


class InsEmb_TPD_579_ALL_PRODataset(DigitsDataset):
    
    def Splitdata(self, patient_all, label_all, protein_all, set_all, splitstr):

        choose_bool = [True if set_item == splitstr else False for set_item in set_all]
        # print(choose_bool)
        
        patient_new = []
        label_new = []

        for i in range(len(set_all)):
            if choose_bool[i] == True:
                patient_new.append(patient_all[i])
                label_new.append(label_all[i])

        protein_new = protein_all[choose_bool,:]
        return patient_new, label_new, protein_new

    def __init__(self, data_name="InsEmb_TPD_579", train=True, datapath="~/data"):
        self.data_name = data_name

        data_frame = pd.read_csv(datapath+'/alldata.csv')
        patient_all = data_frame['Unnamed: 0'].tolist()
        label_all_str = data_frame['label'].tolist()
        label_all = [1.0 if item == 'B' else 0.0 for item in label_all_str]
        set_all = data_frame['Sets'].tolist()
        protein_all = data_frame.drop(['Unnamed: 0', 'label', 'Sets'], axis=1).to_numpy()
        protein_name = data_frame.columns.to_list()
        protein_name.remove('label')
        protein_name.remove('Unnamed: 0')
        protein_name.remove('Sets')
        patient_dis, label_dis, protein_dis = self.Splitdata(patient_all, label_all, protein_all, set_all, 'Discovery')
        # patient_ret, label_ret, protein_ret = self.Splitdata(patient_all, label_all, protein_all, set_all, 'Retrospective Test')
        patient_dis = np.concatenate([patient_dis])
        label_dis = np.concatenate([label_dis])
        protein_dis = np.concatenate([protein_dis])
        # protein_list = [
        #     "P02765","P04083","O00339","P58546","O75347","P04216","P02751",
        #     "P83731","P00568","P78527","P04792","P57737","P42224","P27797",
        #     "Q9HAT2","P30086","O14964","P10909","P17931"
        #     ]
        # index_list = Get_protein_index(protein_name, protein_list)
        index_list = [i for i in range(protein_dis.shape[1])]
        data_dis = Get_selected_protein_matrix(protein_dis, index_list)
        mask_nan = np.isnan(data_dis).sum(axis=0)<20
        data_dis = Fill_nan(data_dis, 12)
        data = tensor(data_dis).float()[:, mask_nan]
        label = tensor(label_dis)
        
        self.def_fea_aim = 50
        self.index_s = list(range(data.shape[1]))
        self.train_val_split(data, label, train, split_int=4)
        self.graphwithpca = False


class InsEmb_TPD_579_ALL_PRO5CDataset(DigitsDataset):
    
    def Splitdata(self, patient_all, label_all, protein_all, set_all, splitstr):

        choose_bool = [True if set_item == splitstr else False for set_item in set_all]
        # print(choose_bool)
        
        patient_new = []
        label_new = []

        for i in range(len(set_all)):
            if choose_bool[i] == True:
                patient_new.append(patient_all[i])
                label_new.append(label_all[i])

        protein_new = protein_all[choose_bool,:]
        return patient_new, label_new, protein_new

    def __init__(self, data_name="InsEmb_TPD_579", train=True, datapath="~/data"):
        self.data_name = data_name

        data_frame = pd.read_csv(datapath+'/alldata_Histopathology_type_list01204_class_5.csv')
        patient_all = data_frame['Unnamed: 0'].tolist()
        label_all_str = data_frame['Histopathology_type'].tolist()

        label_name_list = list(set(data_frame['Histopathology_type'].tolist()))

        label_all = [label_name_list.index(item) for item in label_all_str]
        set_all = data_frame['Sets'].tolist()
        protein_all = data_frame.drop(['Unnamed: 0', 'label', 'Sets','Histopathology_type'], axis=1).to_numpy()
        protein_name = data_frame.columns.to_list()
        # protein_name.remove('label')
        protein_name.remove('Histopathology_type')
        protein_name.remove('label')
        protein_name.remove('Unnamed: 0')
        protein_name.remove('Sets')
        patient_dis, label_dis, protein_dis = self.Splitdata(patient_all, label_all, protein_all, set_all, 'Discovery')
        # patient_ret, label_ret, protein_ret = self.Splitdata(patient_all, label_all, protein_all, set_all, 'Retrospective Test')
        patient_dis = np.concatenate([patient_dis])
        label_dis = np.concatenate([label_dis])
        protein_dis = np.concatenate([protein_dis])
        # protein_list = [
        #     "P02765","P04083","O00339","P58546","O75347","P04216","P02751",
        #     "P83731","P00568","P78527","P04792","P57737","P42224","P27797",
        #     "Q9HAT2","P30086","O14964","P10909","P17931"
        #     ]
        # index_list = Get_protein_index(protein_name, protein_list)
        index_list = [i for i in range(protein_dis.shape[1])]
        data_dis = Get_selected_protein_matrix(protein_dis, index_list).astype(np.float32)
        mask_nan = np.isnan(data_dis).sum(axis=0) < 20
        data_dis = Fill_nan(data_dis, 12)
        data = tensor(data_dis).float()[:, mask_nan]
        label = tensor(label_dis)
        
        self.feature_name = np.array(protein_name)[mask_nan]
        self.label_name_list = label_name_list
        self.def_fea_aim = 50
        self.index_s = list(range(data.shape[1]))
        self.train_val_split(data, label, train, split_int=4)
        self.graphwithpca = False

class InsEmb_PBMCDataset(DigitsDataset):
    def __init__(self, data_name="InsEmb_PBMC", train=True, datapath="~/data"):
        self.data_name = data_name
        # data_raw = pd.read_csv(datapath+'/feature_emb/univ.csv').drop(['Name'], axis=1).to_numpy()
        adata = sc.read(datapath+"/PBMC3k_HVG_regscale.h5ad")
        # data = tensor(np.array(adata.X)).float()
        data = tensor(adata.obsm['X_pca'].copy()).float()
        label_train_str = list(adata.obs['celltype'])
        label_train_str_set = list(set(label_train_str))
        label = tensor([label_train_str_set.index(i) for i in label_train_str])
        
        self.def_fea_aim = 50
        index_s = [33,27,48,3,22,14,30,38,1,7,42,8,47,5,4,2]
        self.index_s = index_s
        self.train_val_split(data, label, train, split_int=5)
        self.graphwithpca = False

class InsEmb_ColonDataset(DigitsDataset):
    def __init__(self, data_name="InsEmb_Colon", train=True, datapath="~/data"):
        self.data_name = data_name
        index_s = [9,2,3,48,236,17,27,81,91,42,52,69,44,232,286,28]
        # data_raw = pd.read_csv(datapath+'/feature_emb/univ.csv').drop(['Name'], axis=1).to_numpy()
        sadata = sc.read(datapath+"/colonn.h5ad")
        data = tensor(sadata.X)
        # data_show = data[:, index_s]
        label = tensor(
            np.array([int(i) for i in list(sadata.obs.clusterstr)])
            )
        
        self.index_s = index_s
        self.def_fea_aim = 50
        self.train_val_split(data, label, train, split_int=5)
        self.graphwithpca = False

class InsEmb_DigitDataset(DigitsDataset):
    def __init__(self, data_name="InsEmb_Digit", train=True, datapath="~/data"):
        self.data_name = data_name
        # data_raw = pd.read_csv(datapath+'/feature_emb/univ.csv').drop(['Name'], axis=1).to_numpy()
        digit = load_digits()
        data = tensor(digit.data).float()/16
        # feature_std = torch.std(data, dim=0)
        # data = data[:, feature_std>feature_std.mean()]
        label = tensor(digit.target)
        
        self.def_fea_aim = 50
        self.index_s = [28,42,52,37,53,20,26,27,5,36,59,2,62,41,6,30]
        self.train_val_split(data, label, train, split_int=5)
        self.graphwithpca = False

class InsEmb_MnistDataset(DigitsDataset):
    def __init__(self, data_name="Mnist", train=True, datapath="~/data"):
        # digit = load_digits()
        self.data_name = data_name
        D = datasets.MNIST(root=datapath, train=True,
                           download=True, transform=None)

        data = (D.data[:20000] / 255).float().reshape((20000, -1))
        label = (D.targets[:20000]).reshape((-1))

        self.def_fea_aim = 50
        self.train_val_split(data, label, train)
        self.graphwithpca = False