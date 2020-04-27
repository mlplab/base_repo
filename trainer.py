# coding: utf-8


import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
import torch
from utils import psnr


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer(object):

    def __init__(self, model, criterion, optimizer, scheduler=None, callbacks=None):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
        #                                                             mode='min',
        #                                                             factor=.2,
        #                                                             patience=2,
        #                                                             verbose=True,
        #                                                             min_lr=1e-8)
        self.callbacks = callbacks

    def train(self, epochs, train_dataloader, val_dataloader, init_epoch=None):

        if init_epoch is None:
            init_epoch = 0
        elif isinstance(init_epoch, int):
            assert 'Please enter int to init_epochs'

        # _, columns = os.popen('stty size', 'r').read().split()
        # columns = int(columns) // 2
        columns = 200

        for epoch in range(init_epoch, epochs):
            dt_now = datetime.now()
            print(dt_now)
            self.model.train()
            mode = 'Train'
            train_loss = []
            val_loss = []
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            with tqdm(train_dataloader, desc=desc_str, ncols=columns, unit='step') as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = self._trans_data(inputs, labels)
                    loss = self._step(inputs, labels)
                    train_loss.append(loss.item())
                    psnr_show = psnr(loss)
                    self._step_show(pbar, Loss=f'{loss:.7f}', PSNR=f'{psnr_show:.7f}')
                    torch.cuda.empty_cache()
            mode = 'Val'
            self.model.eval()
            desc_str = f'{mode:>5} Epoch: {epoch + 1:05d} / {epochs:05d}'
            with tqdm(val_dataloader, desc=desc_str, ncols=columns, unit='step') as pbar:
                for i, (inputs, labels) in enumerate(pbar):
                    inputs, labels = self._trans_data(inputs, labels)
                    with torch.no_grad():
                        loss = self._step(inputs, labels, train=False)
                    val_loss.append(loss.item())
                    psnr_show = psnr(loss)
                    self._step_show(pbar, Loss=f'{loss:.7f}', PSNR=f'{psnr_show:.7f}')
                    torch.cuda.empty_cache()
            train_loss = np.mean(train_loss)
            val_loss = np.mean(val_loss)
            if self.callbacks:
                for callback in self.callbacks:
                    callback.callback(self.model, epoch, loss=train_loss,
                                      val_loss=val_loss, save=True, device=device)
            if self.scheduler is not None:
                self.scheduler.step()
            print('-' * int(columns))

        return self

    def _trans_data(self, inputs, labels):
        inputs = inputs.to(device)
        labels = labels.to(device)
        return inputs, labels

    def _step(self, inputs, labels, train=True):
        if train is True:
            self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, labels)
        if train is True:
            loss.backward()
            self.optimizer.step()
        return loss

    # def _step_show(self, pbar, mode, epoch, loss, psnr_show):
    #     if device == 'cuda':
    #         pbar.set_postfix(
    #             OrderedDict(
    #                 Loss=f'{loss:.7f}',
    #                 PSNR=f'{psnr_show:.7f}',
    #                 Allocate=f'{torch.cuda.memory_allocated(0) / 1024 ** 3:.3f}GB',
    #                 Cache=f'{torch.cuda.memory_cached(0) / 1024 ** 3:.3f}GB'
    #             )
    #         )
    #     elif device == 'cpu':
    #         pbar.set_postfix(
    #             OrderedDict(
    #                 Loss=f'{loss:.7f};',
    #                 PSNR=f'{psnr_show:.7f}'
    #             )
    #         )
    def _step_show(self, pbar, *args, **kwargs):
        if device == 'cuda':
            kwargs['Allocate']=f'{torch.cuda.memory_allocated(0) / 1024 ** 3:.3f}GB'
            kwargs['Cache']=f'{torch.cuda.memory_cached(0) / 1024 ** 3:.3f}GB'
        pbar.set_postfix(kwargs)
        return self


class Deeper_Trainer(Trainer):

    def _step(self, inputs, labels, train=True):
        if train is True:
            self.optimizer.zero_grad()
        output_6, output_12, output = self.model(inputs)
        labels_6 = labels[:, ::4]
        labels_12 = labels[:, ::2]
        loss = .1 * self.criterion(output_6, labels_6) + .1 * self.criterion(output_12, labels_12) + self.criterion(output, labels)
        # loss = self.criterion(output, labels)
        if train is True:
            loss.backward()
            self.optimizer.step()
        show_loss = torch.nn.functional.mse_loss(output, labels)
        return show_loss
