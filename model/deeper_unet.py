# coding: utf-8


import torch
import torchvision
from torchsummary import summary
from .base_model import Conv_Block


class Deeper_UNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature_list=None):
        super(Deeper_UNet, self).__init__()

        if feature_list is None:
            feature_list = [64, 128, 256, 512, 1024]
        self.start_conv = Conv_Block(input_ch, feature_list[0], 1, 1, 0)
        layer_num = 2
        encode_block = []
        len_feature_list = len(feature_list)
        for i in range(len_feature_list - 1):
            layer_block = [Conv_Block(
                feature_list[i], feature_list[i + 1], 3, 1, 1)]
            layer_block = layer_block + \
                [Conv_Block(feature_list[i + 1], feature_list[i + 1], 3, 1, 1)
                 for _ in range(layer_num)]
            # for _ in range(layer_num):
            #     layer_block.append(
            #         Conv_Block(feature_list[i + 1], feature_list[i + 1], 3, 1, 1)
            #         )
            if i < len(feature_list) - 2:
                layer_block.append(torch.nn.MaxPool2d(2))
            layer_block = torch.nn.Sequential(*layer_block)
            encode_block.append(layer_block)
        self.encode_block = torch.nn.Sequential(*encode_block)

    def forward(self, x):

        x = self.start_conv(x)
        encode_block = [x]
        for i, encode in enumerate(self.encode_block):
            x = encode(x)
            encode_block.append(x)
        for i, block in enumerate(encode_block):
            print(i, block.shape)

        return x


if __name__ == '__main__':

    model = Deeper_UNet(25, 24).to('cpu')
    x = torch.rand((1, 25, 96, 96)).to('cpu')
    y = model(x)
    summary(model, (25, 96, 96))
