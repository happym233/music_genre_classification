import torch
import torch.nn as nn
from .CNN_block import CNN_2d_block, ResBlock
from .MLP import MLP

"""
    two 2d convolutional block => one LSTM block => one MLP block
    output_dim: int
        the output dimension of the model
    LSTM_hidden_size: int
        the hidden size of LSTM
    bidirectional: boolean
        True of apply bidirectional LSTM
"""


class MusicCRDNN(nn.Module):

    def __init__(self, CNN_out_channels=[], output_dim=10, LSTM_input_size=80, LSTM_hidden_size=240,
                 MLP_hidden_dims=[60], res_block=False, bidirectional=False):
        super(MusicCRDNN, self).__init__()

        CNN_block_array = []
        cur = 1
        for CNN_out_channel in CNN_out_channels:
            CNN_block_array.append(CNN_2d_block(
                in_channels=cur,
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
                CNN_block_array.append(ResBlock(channels=CNN_out_channel))
            cur = CNN_out_channel

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


