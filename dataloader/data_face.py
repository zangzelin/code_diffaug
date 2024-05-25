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

import os

# import cv2
import numpy as np
import logging
import os
import shutil

from matplotlib import cm
from sklearn.model_selection import train_test_split
import pandas as pd
#from ggplot import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def read_images_from_single_face_profile(face_profile, face_profile_name_index, dim = (48, 48)):
                     # face_profile: ../face_profiles/yaleBnn
    """
    Reads all the images from one specified face profile into ndarrays
    Parameters
    ----------
    face_profile: string
        The directory path of a specified face profile
    face_profile_name_index: int
        The name corresponding to the face profile is encoded in its index
    dim: tuple = (int, int)
        The new dimensions of the images to resize to
    Returns
    -------
    X_data : numpy array, shape = (number_of_faces_in_one_face_profile, face_pixel_width * face_pixel_height)
        A face data array contains the face image pixel rgb values of all the images in the specified face profile
    Y_data : numpy array, shape = (number_of_images_in_face_profiles, 1)
        A face_profile_index data array contains the index of the face profile name of the specified face profile directory
    """
    X_data = np.array([])
    index = 0
    for the_file in os.listdir(face_profile):    #face_profile: ../face_profiles/yaleBnn
        file_path = os.path.join(face_profile, the_file)
        if file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith(".jpeg") or file_path.endswith(".pgm"):
            img = cv2.imread(file_path, 0)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            img_data = img.ravel()
            X_data = img_data if not X_data.shape[0] else np.vstack((X_data, img_data))
            index += 1

    if index == 0 :
        shutil.rmtree(face_profile)
        logging.error("\nThere exists face profiles without images")

    Y_data = np.empty(index, dtype = int)                        #number of pictures in one yaleB file
    Y_data.fill(face_profile_name_index)    # Y_data: [face_profile_name_index,......,face_profile_name_index ]
                                                    # [i,i,i,i,..........................................i,i,i]
    return X_data, Y_data 


def load_training_data(face_profile_directory):  #face_profile_directory   ../face_profiles/

    """
    Loads all the images from the face profile directory into ndarrays
    Parameters
    ----------
    face_profile_directory: string
        The directory path of the specified face profile directory
    face_profile_names: list
        The index corresponding to the names corresponding to the face profile directory
    Returns
    -------
    X_data : numpy array, shape = (number_of_faces_in_face_profiles, face_pixel_width * face_pixel_height)
        A face data array contains the face image pixel rgb values of all face_profiles
    Y_data : numpy array, shape = (number_of_face_profiles, 1)
        A face_profile_index data array contains the indexs of all the face profile names
    """

    # Get a the list of folder names in face_profile as the profile names
    face_profile_names = [d for d in os.listdir(face_profile_directory) if "." not in str(d)]
    # face_profile_names :yaleB01,yaleB02......
    # if len(face_profile_names) < 2:
    #     logging.error("\nFace profile contains too little profiles (At least 2 profiles are needed)")
    #     exit()

    first_data = str(face_profile_names[0])
    first_data_path = os.path.join(face_profile_directory, first_data)   # first_data_path:../face_profiles/yaleB01
    X1, y1 = read_images_from_single_face_profile(first_data_path, 0)
    X_data = X1
    Y_data = y1
    print ("Loading Database: ")
    print (0, "    ",X1.shape[0]," images are loaded from:", first_data_path)
    for i in range(1, len(face_profile_names)):
        directory_name = str(face_profile_names[i])
        directory_path = os.path.join(face_profile_directory, directory_name)
        tempX, tempY = read_images_from_single_face_profile(directory_path, i)
        X_data = np.concatenate((X_data, tempX), axis=0)
        Y_data = np.append(Y_data, tempY)
        print (i, "    ",tempX.shape[0]," images are loaded from:", directory_path)

    return X_data, Y_data, face_profile_names       # X_data: (2452,2500), Y_data: (2452,)



class InsEmb_FaceYaleBDataset(DigitsDataset):
    def __init__(self, data_name="InsEmb_FaceYaleB", train=True, datapath="~/data"):
        self.data_name = data_name
        # data_raw = pd.read_csv(datapath+'/feature_emb/car2.csv').drop(['Name'], axis=1).to_numpy()
        # data = tensor(StandardScaler().fit_transform(data_raw)).float()
        # label = tensor([0]*data.shape[0])
        
        data , label , _ = load_training_data(datapath+"/Datasets_dr/ExtyaleB/CroppedYale")
        data = torch.tensor(data).float()
        label = torch.tensor(label).float()


        self.def_fea_aim = 50
        self.index_s = list(range(data.shape[1]))
        self.train_val_split(data, label, train, split_int=5)
        self.graphwithpca = False
