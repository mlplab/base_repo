# coding


import os
import torch
import torchvision
import numpy as np
import scipy.io as sio
from utils import normalize


data_path = 'dataset/'
size = 256


class HyperSpectralDataset(torch.utils.data.Dataset):

    def __init__(self, img_path, mask_path, concat=False, tanh=False, transform=None):

        self.img_path = img_path
        self.data = os.listdir(img_path)
        # self.mask = sio.loadmat(os.path.join(img_path, mask_path))
        mask = sio.loadmat(mask_path)['data'].astype(np.float32)
        self.mask = torchvision.transforms.ToTensor()(mask)
        self.data_len = len(self.data)
        self.tanh = tanh
        self.concat = concat
        self.transforms = transform

    def __getitem__(self, idx):
        mat_data = sio.loadmat(os.path.join(self.img_path, self.data[idx]))['data']
        # mat_data = mat_data['data']
        nd_data = np.array(mat_data, dtype=np.float32).copy()
        if self.transforms is not None:
            for transform in self.transforms:
                nd_data = transform(nd_data)
        else:
            nd_data = torchvision.transforms.ToTensor()(nd_data)
        trans_data = nd_data
        # label_data = normalize(trans_data)
        label_data = trans_data / trans_data.max()
        # trans_data = torchvision.transforms.ToTensor()(nd_data)
        measurement_data = torch.sum(trans_data * self.mask, dim=0).unsqueeze(0)
        # measurement_data = (measurement_data - measurement_data.min()) / (measurement_data.max() - measurement_data.min())
        # measurement_data = normalize(measurement_data)
        measurement_data = measurement_data / measurement_data.max()
        if self.tanh is True:
            label_data = label_data * 2. - 1.
            measurement_data = measurement_data * 2. - 1.
        if self.concat is True:
            input_data = torch.cat([measurement_data, self.mask], dim=0)
        else:
            input_data = measurement_data
        return input_data, label_data

    def __len__(self):
        return self.data_len


class RefineEvaluateDataset_Random(torch.utils.data.Dataset):

    def __init__(self, img_path, label_path, band=24, transform=None):

        self.img_path = img_path
        self.img_data = os.listdir(img_path)
        self.label_path = label_path
        self.label_data = os.listdir(label_path)
        self.data_len = len(self.img_data)
        self.band = band
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.norm = torchvision.transforms.Normalize((.5,), (.5,))

    def __getitem__(self, idx):

        # choice_band = np.random.choice(self.band)
        img_data = sio.loadmat(os.path.join(self.img_path, self.img_data[idx]))['data'][:, :, self.band]
        nd_data = np.array(img_data, dtype=np.float32)
        # nd_data = torchvision.transforms.ToTensor()(nd_data)
        input_data = self.norm(self.to_tensor(nd_data))
        label_data = sio.loadmat(os.path.join(self.label_path, self.label_data[idx]))['data'][:, :, self.band]
        label_data = np.array(label_data, dtype=np.float32)
        label_data = self.norm(self.to_tensor(label_data))
        return input_data, label_data

    def __len__(self):
        return self.data_len


class RefineEvaluateDataset(torch.utils.data.Dataset):

    def __init__(self, img_path, label_path, transform=None, tanh=False):

        self.img_path = img_path
        self.img_data = os.listdir(img_path)
        self.label_path = label_path
        self.label_data = os.listdir(label_path)
        self.data_len = len(self.img_data)
        self.transform = transform
        self.tanh = tanh
        self.to_tensor = torchvision.transforms.ToTensor()
        self.norm = torchvision.transforms.Normalize((.5,), (.5,))

    def __getitem__(self, idx):

        img_data = sio.loadmat(os.path.join(self.img_path, self.img_data[idx]))['data']
        nd_data = np.array(img_data, dtype=np.float32)
        input_data = self.to_tensor(nd_data)
        if self.tanh is True:
            input_data = self.norm(input_data)
        label_data = sio.loadmat(os.path.join(self.label_path, self.label_data[idx]))['data']
        label_data = np.array(label_data, dtype=np.float32)
        label_data = self.to_tensor(label_data)
        if self.tanh is True:
            label_data = self.norm(label_data)
        return input_data, label_data

    def __len__(self):
        return self.data_len
