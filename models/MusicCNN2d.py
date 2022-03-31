import torch
import torch.nn as nn
from .MLP import MLP
from .CNN_block import CNN_2d_block


class MusicCNN2d(nn.Module):

    def __init__(self):
        super(MusicCNN2d, self).__init__()
        self.conv1 = CNN_2d_block(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=1,
            pooling_kernel_size=2,
            pooling_stride=2,
            pooling_padding=0,
            pooling='max',
            activation='ReLU',
            batch_norm=True
        )

        self.MLP = MLP(10000, 4, [1000, 50])
        # self.linear3 = nn.Linear(50, 4)

    def forward(self, x):
        out = self.conv1(x)
        # print(out.shape)
        out = torch.flatten(out, 1)
        out = self.MLP(out)
        return out