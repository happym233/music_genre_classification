import torch
import torch.nn as nn
from .CNN_block import CNN_1d_block
from .MLP import MLP


class MusicCNN1d(nn.Module):

    """
        1d CNN for music genre classification
        several fixed CNN blocks(different output channels) followed by a DNN block

        CNN_out_channels: list
            an array of output channels
        pooling: str
            'avg' if use average pooling, 'max' if use max pooling

        DNN: DNN block following the CNN network

        DNN_input_dim: int
            input dimension of DNN block
        DNN_output_dim: int
            output dimension of DNN block
        DNN_hidden_dims: list
            hidden dimensions of DNN clock
    """
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
