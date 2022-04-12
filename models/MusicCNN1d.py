import torch
import torch.nn as nn
from .CNN_block import CNN_1d_block, Res1d
from .MLP import MLP


class MusicCNN1d(nn.Module):
    """
        1d CNN for music genre classification
        several fixed CNN blocks(different output channels) followed by a DNN block

        Arguments
        ---------
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

        Example
        ---------
        >>>model = MusicCNN1d(CNN_input_channels=1,
        >>>           CNN_out_channels=[2, 4, 8],
        >>>           pooling='avg',
        >>>           DNN_input_dim=8,
        >>>           DNN_output_dim=10,
        >>>           DNN_hidden_dims=[50],
        >>>           res_block=False)
        >>>inp_tensor = torch.rand([10, 1, 100])
        >>>out_tensor = model(inp_tensor)
        >>>print(out_tensor.shape)
        torch.Size([10, 10])
    """

    def __init__(self, CNN_input_channels=1, CNN_out_channels=None, pooling='avg', DNN_input_dim=26720, DNN_output_dim=10,
                 DNN_hidden_dims=[], res_block=False):

        super(MusicCNN1d, self).__init__()

        if CNN_out_channels is None:
            raise Exception('Empty CNN out channels')
        input_channels = CNN_input_channels
        CNN_block_array = []
        for CNN_out_channel in CNN_out_channels:
            CNN_block_array.append(CNN_1d_block(
                in_channels=input_channels,
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

            if res_block:
                CNN_block_array.append(Res1d(CNN_out_channel))
            input_channels = CNN_out_channel
            # print(cur)

        self.CNN = nn.Sequential(*CNN_block_array)
        self.MLP = MLP(DNN_input_dim, DNN_output_dim, DNN_hidden_dims)

    def forward(self, x):
        out = self.CNN(x)
        out = torch.flatten(out, 1)
        out = self.MLP(out)
        return out
