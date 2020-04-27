# coding: utf-8

import os
import torch
import torchvision
from trainer import Trainer
from gan_trainer import GAN_Trainer
from model.unet import UNet
from model.discriminator import Discriminator
from data_loader import HyperSpectralDdataset
from utils import RandomCrop, RandomHorizontalFlip, ModelCheckPoint


crop = 256
batch_size = 1
epochs = 5
# data_len = batch_size * 10

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

img_path = 'dataset/'
train_path = os.path.join(img_path, 'train')
test_path = os.path.join(img_path, 'test')
# drive_path = '/content/drive/My Drive/auto_colorization/'

train_list = os.listdir(train_path)
test_list = os.listdir(test_path)
transform = (RandomCrop(crop), RandomHorizontalFlip(),
             torchvision.transform.ToTensor())
test_transform = (RandomHorizontalFlip(), torchvision.transforms.ToTensor())
train_dataset = HyperSpectralDdataset(
    train_path, os.path.join(img_path, 'mask.mat'), transform=transform)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataset = HyperSpectralDdataset(
    test_path, os.path.join(img_path, 'mask.mat'), transform=test_transform)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
g_model = UNet(25, 24).to(device)
d_model = Discriminator([32, 64, 128], (crop, crop), 25).to(device)
g_loss = [torch.nn.MSELoss().to(device), torch.nn.BCELoss().to(device)]
d_loss = torch.nn.BCELoss().to(device)
g_param = list(g_model.parameters())
d_param = list(d_model.parameters())
g_optim = torch.optim.Adam(lr=1e-3, params=g_param)
d_optim = torch.optim.Adam(lr=1e-3, params=d_param)
g_ckpt_cb = ModelCheckPoint('g_ckpt', 'unet')
d_ckpt_cb = ModelCheckPoint('d_ckpt', 'discriminator')
trainer = GAN_Trainer(g_model, d_model, g_loss, d_loss, g_optim, d_optim, batch_size,
                      device=device, g_callbacks=[g_ckpt_cb], d_callbacks=[d_ckpt_cb])
trainer.train(epochs, train_dataloader, None)
