# # coding: utf-8


import torch
from torchsummary import summary
from base_model import Conv_Block, D_Conv_Block, DenseBlock, TransBlock


class Dense_UNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, first_feature, k, mode='RGB'):
        super(Dense_UNet, self).__init__()

        feature_list = [first_feature]

        encoder = []
        self.pool = torch.nn.MaxPool2d(2)
        self.input_conv = Conv_Block(1, first_feature, 3, 1, 1, norm=False)
        for i in range(len(k)):
            encoder.append(DenseBlock(feature_list[i], k[i], layer_num=4))
            feature_list.append(feature_list[i] + (k[i] * 4))
        self.encoder = torch.nn.Sequential(*encoder)

        decode_feature = feature_list[::-1]
        decode_k = k[::-1][1:]
        up_conv = []
        pointwise = []
        decoder = []
        for i in range(len(decode_k)):
            up_conv.append(D_Conv_Block(decode_feature[i], decode_feature[i + 1], norm=False))
            # pointwise.append(Conv_Block(decode_feature[i + 1] * 2, decode_feature[i + 1] // 2, 1, 1, 0))
            # decoder.append(DenseBlock(decode_feature[i + 1] // 2, decode_k[i], layer_num=4))
            pointwise.append(Conv_Block(decode_feature[i + 1] * 2, decode_feature[i + 2], 1, 1, 0, norm=False))
            decoder.append(DenseBlock(decode_feature[i + 2], decode_k[i], layer_num=4))
        self.up_conv = torch.nn.Sequential(*up_conv)
        self.pointwise = torch.nn.Sequential(*pointwise)
        self.decoder = torch.nn.Sequential(*decoder)
        if mode is 'RGB':
            activation = torch.nn.Sigmoid()
        elif mode is 'YCbCr':
            activation = torch.nn.Tanh()

        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(decode_feature[-2], output_ch, kernel_size=1, padding=0),
            activation
        )
    def forward(self, x):
        encode = []
        x = self.input_conv(x)
        for i in range(len(self.encoder) - 1):
            x = self.encoder[i](x)
            encode.append(x)
            x = self.pool(x)
        x = self.encoder[-1](x)
        encode = encode[::-1]

        for i, (up_conv, pointwise, decoder) in enumerate(zip(self.up_conv, self.pointwise, self.decoder)):
            x = up_conv(x)
            x = torch.cat((x, encode[i]), dim=1)
            x = pointwise(x)
            x = decoder(x)
        x = self.output(x)
        return x


if __name__ == '__main__':

    print('start code')
    model = Dense_UNet(1, 3, 64, [1, 2, 4, 3])
    print('load model')
    # model = Dense_UNet(1, 3, 32, [8, 16, 32, 64, 128])
    summary(model, (1, 128, 128))
    # torch.save(model.state_dict(), 'dense_unet.pth')