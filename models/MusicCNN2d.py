import torch
import torch.nn as nn
from .MLP import MLP
from .CNN_block import CNN_2d_block


class MusicCNN2d_1CNNBlock(nn.Module):
    '''
        one 2d CNN block followed by one DNN block
    '''
    def __init__(self, output_dim, out_channels, dnn_input_dim):
        super(MusicCNN2d_1CNNBlock, self).__init__()
        self.conv1 = CNN_2d_block(
            in_channels=1,
            out_channels=out_channels,
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

        self.MLP = MLP(dnn_input_dim, output_dim=output_dim, hidden_dims=[5000, 200])
        # self.linear3 = nn.Linear(50, 4)

    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        out = torch.flatten(out, 1)
        out = self.MLP(out)
        return out


class MusicCNN2d_2CNNBlock(nn.Module):

    def __init__(self, out_channel1=8, out_channel2=32, DNN_input_dim=10000, DNN_hidden_dims=[],  DNN_output_dim=10):
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
