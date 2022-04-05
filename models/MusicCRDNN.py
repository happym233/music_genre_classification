import torch
import torch.nn as nn
from .CNN_block import CNN_2d_block
from .MLP import MLP


class MusicCRDNN(nn.Module):
    def __init__(self, output_dim):
        super(MusicCRDNN, self).__init__()

        self.conv1 = CNN_2d_block(
            in_channels=1,
            out_channels=4,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            pooling='avg',
            pooling_kernel_size=(2, 2),
            pooling_stride=(2, 2),
            pooling_padding=0,
            activation='ReLU',
            batch_norm=True

        )

        self.conv2 = CNN_2d_block(
            in_channels=4,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            pooling='avg',
            pooling_kernel_size=(2, 2),
            pooling_stride=(2, 2),
            pooling_padding=0,
            activation='ReLU',
            batch_norm=True

        )

        self.LSTM = nn.LSTM(input_size=80, hidden_size=240, num_layers=8, dropout=0.15, batch_first=True)

        self.MLP = MLP(input_dim=240, output_dim=output_dim, hidden_dim=[60])

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.flatten(torch.transpose(out, 1, 2), 2)
        # print(out.shape)
        out, _ = self.LSTM(out)
        out = torch.mean(out, dim=1)
        # print(out.shape)
        out = self.MLP(out)
        return out
