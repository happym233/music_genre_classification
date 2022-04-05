import torch
import torch.nn as nn
from .CNN_block import CNN_1d_block
from .MLP import MLP


class MusicCNN1d(nn.Module):
    def __init__(self, CNN_out_channels=None, pooling='avg', DNN_input_dim=26720, DNN_output_dim=10,
                 DNN_hidden_dims=[]):

        super(MusicCNN1d, self).__init__()

        if CNN_out_channels is None:
           raise Exception('Empty CNN out channels')
        cur = 1
        CNN_block_array = []
        for CNN_out_channel in CNN_out_channels:
            CNN_block_array.append(CNN_1d_block(
                in_channels=cur,
                out_channels=CNN_out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                pooling_kernel_size=2,
                pooling_stride=2,
                pooling_padding=0,
                pooling=pooling,
                activation='ReLU',
                batch_norm=True
            ))
            cur = CNN_out_channel
            # print(cur)

        self.CNN = nn.Sequential(*CNN_block_array)
        self.MLP = MLP(DNN_input_dim, DNN_output_dim, DNN_hidden_dims)

    def forward(self, x):
        out = self.CNN(x)
        out = torch.flatten(out, 1)
        out = self.MLP(out)
        return out
