# coding: UTF-8


import torch
from torchsummary import summary
from .base_model import Conv_Block, DenseBlock, TransBlock


class DenseNet(torch.nn.Module):

    def __init__(self, input_ch, block_list, k, first_feature=64, compress=.5):
        super(DenseNet, self).__init__()

        # h, w, 3

        # self.conv1 = torch.nn.Conv2d(input_ch, first_feature, (7, 7), stride=2, padding=3)
        self.conv1 = Conv_Block(input_ch, first_feature, 7, 2, 3)
        # h // 2, w // 2, 64
        self.max_pool = torch.nn.MaxPool2d(2)
        # h // 4, w // 4, 64

        ch_list = [first_feature]
        output = []
        dense = []
        trans = []

        for i in range(len(block_list) - 1):
            dense.append(DenseBlock(ch_list[i], k, block_list[i]))
            output.append(ch_list[i] + (k * block_list[i]))
            trans.append(TransBlock(output[i], compress))
            ch_list.append(int(output[i] * compress))
        self.dense = torch.nn.Sequential(*dense)
        self.trans = torch.nn.Sequential(*trans)
        ch_list.append(int(output[-1] * compress))
        self.dense_last = DenseBlock(ch_list[-1], k, block_list[-1])

        '''
        self.dense1 = DenseBlock(first_feature, k, block_list[0])
        output1 = first_feature + (k * block_list[0])
        self.trans1 = TransBlock(output1, compress)
        # h // 8, w // 8, 64 + (k * block_list[0])

        input_ch2 = int(output1 * compress)
        self.dense2 = DenseBlock(input_ch2, k, block_list[1])
        output2 = input_ch2 + (k * block_list[1])
        self.trans2 = TransBlock(output2, compress)
        # h // 16, w // 16

        input_ch3 = int(output2 * compress)
        self.dense3 = DenseBlock(input_ch3, k, block_list[2])
        output3 = input_ch3 + (k * block_list[2])
        self.trans3 = TransBlock(output3, compress)
        # h // 32, w // 32

        input_ch4 = int(output3 * compress)
        self.dense4 = DenseBlock(input_ch4, k, block_list[3])
        '''
        # h // 32, w // 32
        # self.fc = torch.nn.Linear()

    def forward(self, x):
        '''
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)
        '''
        x = self.conv1(x)
        x = self.max_pool(x)
        for dense, trans in zip(self.dense, self.trans):
            x = dense(x)
            x = trans(x)
        x = self.dense_last(x)
        return x


if __name__ == '__main__':

    model = DenseNet(3, [6, 12, 24, 16], k=16, first_feature=16)
    summary(model, (3, 224, 224))
    x = torch.rand((1, 3, 224, 224))
    output = model(x)
    print(output.shape)
