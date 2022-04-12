import torch
import torch.nn as nn
from .CNN_block import CNN_2d_block, Res2d
from .MLP import MLP


class MusicCRDNN(nn.Module):
    '''
        CNN + LSTM + MLP

        Arguments
        ---------
        CNN_out_channels: list
            the number of output channels of CNN blocks
        output_dim: int
            number of classes of classification
        LSTM_input_size: int
            input size of LSTM
        LSTM_hidden_size: int
            hidden size of LSTM
        MLP_hidden_dims: list
            the number of dimensions of hidden layers of MLP
        res_block: boolean
            True if use residual block instead of common CNN block
        bidirectional: boolean
            True if use bidirectional LSTM

        Example
        ---------
        >>>model = MusicCRDNN(CNN_out_channels=[2, 4, 8, 8],
        >>>           output_dim=10,
        >>>           LSTM_input_size=48,
        >>>           LSTM_hidden_size=80,
        >>>           MLP_hidden_dims=[160],
        >>>           res_block=False,
        >>>           bidirectional=False)
        >>>inp_tensor = torch.rand([10, 1, 100, 100])
        >>>out_tensor = model(inp_tensor)
        >>>print(out_tensor.shape)
    '''

    def __init__(self, CNN_out_channels=[], output_dim=10, LSTM_input_size=80, LSTM_hidden_size=240,
                 MLP_hidden_dims=[60], res_block=False, bidirectional=False):
        super(MusicCRDNN, self).__init__()

        CNN_block_array = []
        input_channels = 1
        for CNN_out_channel in CNN_out_channels:
            CNN_block_array.append(CNN_2d_block(
                in_channels=input_channels,
                out_channels=CNN_out_channel,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                pooling='avg',
                pooling_kernel_size=(2, 2),
                pooling_stride=(2, 2),
                pooling_padding=0,
                activation='ReLU',
                batch_norm=True
            ))
            if res_block:
                CNN_block_array.append(Res2d(channels=CNN_out_channel))
            input_channels = CNN_out_channel

        self.conv = nn.Sequential(*CNN_block_array)

        self.LSTM = nn.LSTM(input_size=LSTM_input_size, hidden_size=LSTM_hidden_size,
                            num_layers=8, dropout=0.15, batch_first=True, bidirectional=bidirectional)

        self.MLP = None
        if not bidirectional:
            self.MLP = MLP(input_dim=LSTM_hidden_size, output_dim=output_dim, hidden_dims=MLP_hidden_dims)
        else:
            self.MLP = MLP(input_dim=LSTM_hidden_size * 2, output_dim=output_dim, hidden_dims=MLP_hidden_dims)

    def forward(self, x):
        out = x
        out = self.conv(out)
        out = torch.flatten(torch.transpose(out, 1, 2), 2)
        # print(out.shape)
        out, _ = self.LSTM(out)
        out = torch.mean(out, dim=1)
        # print(out.shape)
        out = self.MLP(out)
        return out


