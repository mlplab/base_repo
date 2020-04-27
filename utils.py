# coding: utf-8


import os
import shutil
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_patch(data_path, save_path, size=256, ch=24, data_key='data'):

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    data_list = os.listdir(data_path)
    for i, name in enumerate(tqdm(data_list)):
        idx = name.split('.')[0]
        f = scipy.io.loadmat(os.path.join(data_path, name))
        data = f[data_key]
        data = np.expand_dims(np.asarray(
            data, np.float32).transpose([2, 0, 1]), axis=0)
        tensor_data = torch.as_tensor(data)
        patch_data = tensor_data.unfold(2, size, size).unfold(3, size, size)
        patch_data = patch_data.permute(
            (0, 2, 3, 1, 4, 5)).reshape(-1, ch, size, size)
        for i in range(patch_data.size()[0]):
            save_data = patch_data[i].to('cpu').detach().numpy().copy().transpose(1, 2, 0)
            save_name = os.path.join(save_path, f'{idx}_{i}.mat')
            scipy.io.savemat(save_name, {'data': save_data})

    return None


def plot_img(output_imgs, title):
    plt.imshow(output_imgs)
    plt.title('Predict')
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.tick_params(length=0)
    return None


def psnr(loss):

    return 20 * torch.log10(1 / torch.sqrt(loss))


class RandomCrop(object):

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        h, w, _ = img.shape
        i = np.random.randint(0, h - self.size[0], dtype=int)
        j = np.random.randint(0, w - self.size[1], dtype=int)
        return img[i: i + self.size[0], j: j + self.size[1], :].copy()


class RandomHorizontalFlip(object):

    def __init__(self, rate=.5):
        if rate:
            self.rate = rate
        else:
            # self.rate = np.random.randn()
            self.rate = .5

    def __call__(self, img):
        if np.random.randn() < self.rate:
            img = img[:, ::-1, :].copy()
        return img


class ModelCheckPoint(object):

    def __init__(self, checkpoint_path, model_name, mkdir=False, partience=1, verbose=True):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.partience = partience
        self.verbose = verbose
        if mkdir is True:
            if os.path.exists(self.checkpoint_path):
                shutil.rmtree(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)

    def callback(self, model, epoch, *args, **kwargs):
        if 'loss' not in kwargs.keys() and 'val_loss' not in kwargs.keys():
            assert 'None Loss'
        else:
            loss = kwargs['loss']
            val_loss = kwargs['val_loss']
        checkpoint_name = os.path.join(self.checkpoint_path, self.model_name +
                                       f'_epoch_{epoch:05d}_loss_{loss:.5f}_valloss_{val_loss:.5f}.pth')
        if epoch % self.partience == 0:
            torch.save(model.state_dict(), checkpoint_name)
            if self.verbose is True:
                print(f'CheckPoint Saved by {checkpoint_name}')
        return self


class Evaluater(object):

    def __init__(self, save_img_path='output_img', save_mat_path='output_mat',
                 save_csv_path='output_csv'):
        self.save_img_path = save_img_path
        self.save_diff_path = os.path.join(save_img_path, 'diff')
        self.save_alls_path = os.path.join(save_img_path, 'alls')
        self.save_output_path = os.path.join(save_img_path, 'output')
        self.save_label_path = os.path.join(save_img_path, 'label')
        self.save_mat_path = save_mat_path
        self.save_csv_path = save_csv_path
        if os.path.exists(save_img_path) is True:
            shutil.rmtree(save_img_path)
        os.mkdir(save_img_path)
        os.mkdir(self.save_diff_path)
        os.mkdir(self.save_alls_path)
        os.mkdir(self.save_label_path)
        os.mkdir(self.save_output_path)
        if os.path.exists(save_mat_path) is True:
            shutil.rmtree(save_mat_path)
        os.mkdir(save_mat_path)

    def _save_img(self, i, inputs, output, labels):
        inputs_plot = inputs[:, 0].unsqueeze(0)
        output_plot = output[:, 10].unsqueeze(0)
        torchvision.utils.save_image(output_plot, os.path.join(self.save_output_path, f'output_{i}.png'))
        label_plot = labels[:, 10].unsqueeze(0)
        torchvision.utils.save_image(label_plot, os.path.join(self.save_label_path, f'label_{i}.png'))
        output_img = torch.cat([inputs_plot, output_plot, label_plot], dim=0)
        torchvision.utils.save_image(output_img, os.path.join(self.save_alls_path, f'out_and_label_{i}.png'), nrow=3, padding=10)
        return self

    def _save_diff(self, i, output, labels):
        _, c, h, w = output.size()
        diff = torch.mean(torch.abs(output - labels), dim=1)
        diff = diff.to('cpu').detach().numpy().copy()
        diff = diff.reshape(h, w)
        plt.imshow(diff, cmap='jet')
        plt.colorbar()
        plt.savefig(os.path.join(self.save_diff_path, f'diff_{i}.png'))
        plt.clf()

    def _save_mat(self, i, output):
        output_mat = output.squeeze().to('cpu').detach().numpy().copy()
        output_mat = output_mat.transpose(1, 2, 0)
        scipy.io.savemat(os.path.join(self.save_mat_path, f'{i}.mat'), {'data': output_mat})
        return self

    def _save_csv(self, output_evaluate, header):
        output_evaluate_np = np.array(output_evaluate)
        means = list(np.mean(output_evaluate_np, axis=0))
        output_evaluate.append(means)
        output_evaluate_csv = pd.DataFrame(output_evaluate)
        output_evaluate_csv.to_csv(self.save_csv_path, header=header)


class ReconstEvaluater(Evaluater):

    def metrics(self, model, dataset, evaluate_fn, header=None, hcr=False):
        model.eval()
        output_evaluate = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(dataset)):
                evaluate_list = []
                inputs = inputs.unsqueeze(0).to(device)
                labels = labels.unsqueeze(0).to(device)
                if hcr is True:
                    _, _, output = model(inputs)
                else:
                    output = model(inputs)
                for metrics_func in evaluate_fn:
                    metrics = metrics_func(output, labels)
                    evaluate_list.append(np.round(metrics.item(), decimals=7))
                output_evaluate.append(evaluate_list)
                self._save_img(i, inputs, output, labels)
                self._save_diff(i, output, labels)
                self._save_mat(i, output)
        self._save_csv(output_evaluate, header)

        return self


# class ReconstEvaluater(Evaluater):
#
#     def metrics(self, model, dataset, evaluate_fn, header=None):
#         model.eval()
#         output_evaluate = []
#         with torch.no_grad():
#             for i, (inputs, labels) in enumerate(tqdm(dataset)):
#                 evaluate_list = []
#                 inputs = inputs.unsqueeze(0).to(device)
#                 labels = labels.unsqueeze(0).to(device)
#                 output = model(inputs)
#                 for metrics_func in evaluate_fn:
#                     metrics = metrics_func(output, labels)
#                     evaluate_list.append(np.round(metrics.item(), decimals=7))
#                 output_evaluate.append(evaluate_list)
#                 self._save_img(i, inputs, output, labels)
#                 self._save_diff(i, output, labels)
#                 self._save_mat(i, output)
#
#         output_evaluate_np = np.array(output_evaluate)
#         means = list(np.mean(output_evaluate_np, axis=0))
#         output_evaluate.append(means)
#         output_evaluate_csv = pd.DataFrame(output_evaluate)
#         output_evaluate_csv.to_csv(os.path.join(
#             self.save_mat_path, 'output_evaluate.csv'), header=header)
#         return self


class RefineEvaluater(Evaluater):

    def __init__(self, ch, save_img_path='output_refine_img',
                 save_mat_path='output_refine_mat', save_csv_path='output_refine_csv'):
        super(RefineEvaluater, self).__init__(save_img_path, save_mat_path,
                                              save_csv_path)
        self.ch = ch

    def metrics(self, model, dataset, evaluate_fn, header=None):
        model.eval()
        output_evaluate = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(dataset)):
                evaluate_list = []
                inputs = inputs.unsqueeze(0).to(device)
                labels = labels.unsqueeze(0).to(device)
                output = model(inputs)
                for metrics_func in evaluate_fn:
                    metrics = metrics_func(output, labels)
                    evaluate_list.append(np.round(metrics.item(), decimals=7))
                output_evaluate.append(evaluate_list)
                self._save_img(i, inputs, output, labels)
                self._save_diff(i, output, labels)
                # self._save_mat(i, output)

        self._save_csv(output_evaluate, header)
        return self

    def _save_img(self, i, inputs, output, labels):
        torchvision.utils.save_image(output, os.path.join(
            self.save_output_path, f'output_{i}.png'))
        torchvision.utils.save_image(labels, os.path.join(
            self.save_label_path, f'label_{i}.png'))
        output_img = torch.cat(
            [inputs, output, labels], dim=0)
        torchvision.utils.save_image(output_img, os.path.join(
            self.save_alls_path, f'out_and_label_{i}.png'), nrow=3, padding=10)
        return self

    def _save_diff(self, i, output, labels):
        _, c, h, w = output.size()
        diff = torch.mean(torch.abs(output - labels),
                          dim=1).to('cpu').detach().numpy().copy()
        diff = diff.reshape(h, w)
        plt.imshow(diff, cmap='jet')
        plt.colorbar()
        plt.savefig(os.path.join(self.save_diff_path, f'diff_{i}.png'))
        plt.clf()


'''

class Draw_Output(object):

    def __init__(self, img_path, output_data, save_path='output', verbose=False, nrow=8):
 Parameters
  ---
   img_path: str
     image dataset path
    output_data: list
     draw output data path
    save_path: str(default: 'output')
     output img path
    verbose: bool(default: False)
     verbose
        self.img_path = img_path
        self.output_data = output_data
        self.data_num = len(output_data)
        self.save_path = save_path
        self.verbose = verbose
        self.input_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])
        self.output_transform = torchvision.transforms.ToPILImage()
        self.nrow = nrow

        ###########################################################
        # Make output directory
        ###########################################################
        if os.path.exists(save_path) is True:
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        if os.path.exists(save_path + '/all_imgs') is True:
            shutil.rmtree(save_path + '/all_imgs')
        os.mkdir(save_path + '/all_imgs')
        ###########################################################
        # Draw Label Img
        ###########################################################
        labels = []
        for data in self.output_data:
            label = self.input_transform(Image.open(os.path.join(self.img_path, data)).convert('RGB'))
            labels.append(label)
        self.labels = torch.cat(labels).reshape(len(labels), *labels[0].shape)
        labels_np = torchvision.utils.make_grid(self.labels, nrow=nrow, padding=10)
        labels_np = labels_np.numpy()
        self.labels_np = np.transpose(labels_np, (1, 2, 0))
        del labels, labels_np
        torchvision.utils.save_image(self.labels, os.path.join(save_path, f'labels.jpg'), nrow=nrow, padding=10)


    def callback(self, model, epoch, *args, **kwargs):
        if 'save' not in kwargs.keys():
            assert 'None save mode'
        else:
            save = kwargs['save']
        device = kwargs['device']
        self.epoch_save_path = os.path.join(self.save_path, f'epoch{epoch}')
        os.makedirs(self.epoch_save_path, exist_ok=True)
        output_imgs = []
        # encoder.eval()
        # decoder.eval()
        for i, data in enumerate(self.output_data):
            img = self.input_transform(Image.open(os.path.join(self.img_path, data)).convert('L')).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img).squeeze().to('cpu')
            output_imgs.append(output)
        output_imgs = torch.cat(output_imgs).reshape(len(output_imgs), *output_imgs[0].shape)
        if self.verbose is True:
            self.__show_output_img_list(output_imgs)
            # self.__show_output_img_list(self.labels)
        if save is True:
            torchvision.utils.save_image(output_imgs, os.path.join(self.save_path, f'all_imgs/all_imgs_{epoch}.jpg'), nrow=self.nrow, padding=10)
        del output_imgs
        return self

    def __draw_output_label(self, output, label, data):
        output = torch.cat((label, output), dim=2)
        output = self.output_transform(output)
        output.save(os.path.join(self.epoch_save_path, data))
        if self.verbose is True:
            print(f'\rDraw Output {data}', end='')
        return self

    def __show_output_img_list(self, output_imgs):
        plt.figure(figsize=(16, 9))
        output_imgs_np = torchvision.utils.make_grid(output_imgs, nrow=self.nrow, padding=10)
        output_imgs_np = output_imgs_np.numpy()
        output_imgs_np = np.transpose(output_imgs_np, (1, 2, 0))
        plt.subplot(1, 2, 1)
        plot_img(output_imgs_np, 'Predict')
        plt.subplot(1, 2, 2)
        plot_img(self.labels_np, 'Label')
        plt.show()
        del output_imgs_np
        return self
'''
