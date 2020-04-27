# coding: utf-8


import torch
from .base_model import Conv_Block_UNet


class Discriminator(torch.nn.Module):

    def __init__(self, feature_list, img_shape, input_ch, layer_num=2):

        super(Discriminator, self).__init__()
        self.input_ch = input_ch
        feature_list.insert(0, input_ch)
        cnn_block = []
        scale = 1
        for i in range(len(feature_list) - 1):
            # cnn_block.append(
            #     CNN_Block(feature_list[i], feature_list[i + 1], num_layer=2))
            cnn_block.append(Conv_Block_UNet(feature_list[i],
                                             feature_list[i + 1], 3, 1, 1))
            for _ in range(layer_num - 1):
                cnn_block.append(Conv_Block_UNet(feature_list[i + 1],
                                                 feature_list[i + 1], 3, 1, 1))
            cnn_block.append(torch.nn.MaxPool2d(2))
            scale *= 2
        self.cnn_block = torch.nn.Sequential(*cnn_block)
        flatten_feature = (img_shape[0] // scale) * \
            (img_shape[1] // scale) * feature_list[-1]
        output_block = [torch.nn.Linear(
            flatten_feature, 1), torch.nn.Sigmoid()]
        self.output_block = torch.nn.Sequential(*output_block)

    def forward(self, x):
        batch_size = x.size()[0]
        cnn_output = self.cnn_block(x)
        flatten_output = cnn_output.view(batch_size, -1)
        disc_output = self.output_block(flatten_output)
        return disc_output
