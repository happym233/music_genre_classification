import torch
import torch.nn as nn


class CNN_1d_block(nn.Module):

    def __init__(
            self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
            pooling='avg', pooling_kernel_size=2, pooling_stride=1, pooling_padding=0,
            activation='ReLU', batch_norm=True
    ):

        super(CNN_1d_block, self).__init__()

        layer_array = []

        conv = nn.Conv1d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding
                         )
        layer_array.append(conv)

        if batch_norm:
            layer_array.append(nn.BatchNorm1d(out_channels))

        if activation.lower() == 'relu' or activation == '':
            layer_array.append(nn.ReLU())
        elif activation.lower() == 'leakyrelu':
            layer_array.append(nn.LeakyReLU)
        else:
            raise Exception('Unknown activation type')

        if pooling.lower() == 'avg':
            layer_array.append(
                nn.AvgPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride, padding=pooling_padding)
            )
        elif pooling.lower() == 'max':
            layer_array.append(
                nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride, padding=pooling_padding)
            )
        else:
            raise Exception('Unknown pooling type')

        self.seq = nn.Sequential(*layer_array)

    def forward(self, x):
        return self.seq(x)


class CNN_2d_block(nn.Module):

    def __init__(
            self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
            pooling='avg', pooling_kernel_size=2, pooling_stride=1, pooling_padding=0,
            activation='relu', batch_norm=True
    ):

        super(CNN_2d_block, self).__init__()

        layer_array = []
        conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding
                         )
        layer_array.append(conv)

        if batch_norm:
            layer_array.append(nn.BatchNorm2d(out_channels))

        if activation.lower() == 'relu' or activation == '':
            layer_array.append(nn.ReLU())
        elif activation.lower() == 'leakyrelu':
            layer_array.append(nn.LeakyReLU(0.01))
        else:
            raise Exception('Unknown activation type')

        if pooling.lower() == 'avg':
            layer_array.append(
                nn.AvgPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride, padding=pooling_padding)
            )
        elif pooling.lower() == 'max':
            layer_array.append(
                nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride, padding=pooling_padding)
            )
        else:
            raise Exception('Unknown pooling type')

        self.seq = nn.Sequential(*layer_array)

    def forward(self, x):
        return self.seq(x)
