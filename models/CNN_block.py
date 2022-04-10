import torch
import torch.nn as nn


class CNN_1d_block(nn.Module):
    """
        1-dimensional CNN block
        CONV1d + activation + batch_norm + pooling
        in_channels: int
            number of input channels
        out_channels: int
            number of output channels
        kernel_size: int
            size of convolutional kernel
        stride: int
            stride of convolutional kernel
        padding: int
            padding of convolution
        pooling: str
            'avg' if use average pooling
            'max' if use max pooling
        pooling_kernel_size: int
            size of pooling kernel
        pooling_stride: int
            stride of pooling kernel
        pooling_padding: int
            padding of pooling kernel
        activation: str
            'ReLU' if use ReLU as activation function
            'LeakyReLU' if use LeakyReLU as activation function
        batch_norm: boolean
            True if use batch norm
            False if not use batch norm
    """

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

        if activation.lower() == 'relu' or activation == '':
            layer_array.append(nn.ReLU())
        elif activation.lower() == 'leakyrelu':
            layer_array.append(nn.LeakyReLU)
        else:
            raise Exception('Unknown activation type')

        if batch_norm:
            layer_array.append(nn.BatchNorm1d(out_channels))

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
    """
        2-dimensional CNN block
        CONV2d + activation + batch_norm + pooling
        in_channels: int
            number of input channels
        out_channels: int
            number of output channels
        kernel_size: int or tuple
            size of convolutional kernel
        stride: int or tuple
            stride of convolutional kernel
        padding: int or tuple
            padding of convolution
        pooling: str
            'avg' if use average pooling
            'max' if use max pooling
        pooling_kernel_size: int or tuple
            size of pooling kernel
        pooling_stride: int or tuple
            stride of pooling kernel
        pooling_padding: int or tuple
            padding of pooling kernel
        activation: str
            'ReLU' if use ReLU as activation function
            'LeakyReLU' if use LeakyReLU as activation function
        batch_norm: boolean
            True if use batch norm
            False if not use batch norm
    """

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

        if activation.lower() == 'relu' or activation == '':
            layer_array.append(nn.ReLU())
        elif activation.lower() == 'leakyrelu':
            layer_array.append(nn.LeakyReLU(0.01))
        else:
            raise Exception('Unknown activation type')

        if batch_norm:
            layer_array.append(nn.BatchNorm2d(out_channels))

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


class Res1d(nn.Module):

    """
        residual convolution block
        conv 3 => batch norm => relu => conv 3 => batch norm => out
                                                                 => x => out + x => relu
        channels: number of channels(unchanged through residual convolutional block)
    """

    def __init__(self, channels):
        super(Res1d, self).__init__()

        self.Conv1d1 = nn.Sequential(
            nn.Conv1d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

        self.Conv1d2 = nn.Sequential(
            nn.Conv1d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        out = self.Conv1d1(x)
        out = self.Conv1d2(out)
        return nn.ReLU()(out + x)

class Res2d(nn.Module):
    '''
        residual convolution block
        conv 3x3 => batch norm => relu => conv 3x3 => batch norm => out
                                                                 => x => out + x => relu
        channels: number of channels(unchanged through residual convolutional block)
    '''

    def __init__(self, channels):
        super(Res2d, self).__init__()

        self.Conv2d1 = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.Conv2d2 = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=channels,
                      kernel_size=3,
                      stride=1,
                      padding=1
                      ),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        out = self.Conv2d1(x)
        out = self.Conv2d2(out)
        return nn.ReLU()(out + x)
