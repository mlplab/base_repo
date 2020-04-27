# coding


import os
import torch
import torchvision
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import utils


data_path = 'dataset/'
size = 256


class HyperSpectralDataset(torch.utils.data.Dataset):

    def __init__(self, img_path, mask_path, transform=None):

        self.img_path = img_path
        self.data = os.listdir(img_path)
        # self.mask = sio.loadmat(os.path.join(img_path, mask_path))
        mask = sio.loadmat(mask_path)['data'].astype(np.float32)
        self.mask = torchvision.transforms.ToTensor()(mask)
        self.data_len = len(self.data)
        self.transforms = transform

    def __getitem__(self, idx):
        mat_data = sio.loadmat(os.path.join(self.img_path, self.data[idx]))['data']
        # mat_data = mat_data['data']
        nd_data = np.array(mat_data, dtype=np.float32)[::-1, :, :].copy()
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = torchvision.transforms.ToTensor()(nd_data)
        trans_data = nd_data
        # trans_data = torchvision.transforms.ToTensor()(nd_data)
        measurement_data = torch.sum(trans_data * self.mask, dim=0).unsqueeze(0)
        input_data = torch.cat([measurement_data, self.mask], dim=0)
        return input_data, trans_data

    def __len__(self):
        return self.data_len


class RefineEvaluateDataset(torch.utils.data.Dataset):

    def __init__(self, img_path, label_path, band=24, transform=None):

        self.img_path = img_path
        self.img_data = os.listdir(img_path)
        self.label_path = label_path
        self.label_data = os.listdir(label_path)
        self.data_len = len(self.img_data)
        self.band = band
        self.transform = transform

    def __getitem__(self, idx):

        # choice_band = np.random.choice(self.band)
        img_data = sio.loadmat(os.path.join(self.img_path, self.img_data[idx]))['data'][:, :, self.band]
        nd_data = np.array(img_data, dtype=np.float32)
        nd_data = torchvision.transforms.ToTensor()(nd_data)
        label_data = sio.loadmat(os.path.join(self.label_path, self.label_data[idx]))['data'][:, :, self.band]
        label_data = np.array(label_data, dtype=np.float32)
        label_data = torchvision.transforms.ToTensor()(label_data)
        return nd_data, label_data

    def __len__(self):
        return self.data_len
a