import torch
import torch.nn as nn
from .MLP import MLP
from .CNN_block import CNN_2d_block, ResBlock


class MusicCNN2d_2CNNBlock(nn.Module):

    """
        2d CNN(2 CNN block) for music genre classification
        2 2d CNN blocks followed by a DNN block

        out_channel1: int
            number of output channels for 1st CNN block
        out_channel2: int
            number of output channels for 2nd CNN block

        DNN: DNN block following the CNN network

        DNN_input_dim: int
            input dimension of DNN block
        DNN_output_dim: int
            output dimension of DNN block
        DNN_hidden_dims: list
            hidden dimensions of DNN clock
    """
    def __init__(self, out_channel1=8, out_channel2=32, DNN_input_dim=10000, DNN_hidden_dims=[], DNN_output_dim=10):
        super(MusicCNN2d_2CNNBlock, self).__init__()
        self.conv1 = CNN_2d_block(
            in_channels=1,
            out_channels=out_channel1,
            kernel_size=3,
            stride=2,
            padding=1,
            pooling_kernel_size=2,
            pooling_stride=2,
            pooling_padding=0,
            pooling='avg',
            activation='ReLU',
            batch_norm=True
        )

        self.conv2 = CNN_2d_block(
            in_channels=out_channel1,
            out_channels=out_channel2,
            kernel_size=3,
            stride=2,
            padding=1,
            pooling_kernel_size=2,
            pooling_stride=2,
            pooling_padding=0,
            pooling='avg',
            activation='ReLU',
            batch_norm=True
        )
        self.MLP = MLP(DNN_input_dim, DNN_output_dim, DNN_hidden_dims)
        # self.linear3 = nn.Linear(50, 4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # print(out.shape)
        out = torch.flatten(out, 1)
        out = self.MLP(out)
        return out


class MusicCNN2d(nn.Module):

    """
        2d CNN for music genre classification
        several fixed CNN blocks(different output channels) followed by a DNN block

        CNN_out_channels: list
            an array of output channels

        DNN: DNN block following the CNN network

        DNN_input_dim: int
            input dimension of DNN block
        DNN_output_dim: int
            output dimension of DNN block
        DNN_hidden_dims: list
            hidden dimensions of DNN clock
    """

    def __init__(self, out_channels=[], DNN_input_dim=10000, DNN_hidden_dims=[], DNN_output_dim=10, res_block=False):
        super(MusicCNN2d, self).__init__()
        cur = 1
        conv_array = []
        for out_channel in out_channels:
            conv_array.append(CNN_2d_block(
                in_channels=cur,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                pooling_kernel_size=2,
                pooling_stride=2,
                pooling_padding=0,
                pooling='avg',
                activation='ReLU',
                batch_norm=True
            ))
            if res_block:
                conv_array.append(ResBlock(out_channel))
            cur = out_channel
        self.conv = nn.Sequential(*conv_array)
        self.MLP = MLP(DNN_input_dim, DNN_output_dim, DNN_hidden_dims)
        # self.linear3 = nn.Linear(50, 4)

    def forward(self, x):
        out = self.conv(x)
        # print(out.shape)
        out = torch.flatten(out, 1)
        out = self.MLP(out)
        return out


