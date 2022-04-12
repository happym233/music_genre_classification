import torch
import torch.nn as nn
from .MLP import MLP


class MusicLSTM(nn.Module):

    """
        LSTM with linear transformation

        Arguments
        ---------
        input_size: int
            size of input features
        hidden_size: int
            size of features in hidden layers
        output_size: int
            size of output of the model
        num_layers: int
            number of recurrent layers
        dropout: float
            dropout rate
        bidirectional: boolean
            True if use bidirectional LSTM

        Example
        ---------
        >>>model = MusicLSTM(input_size=10, hidden_size=20, output_size=10, num_layers=4, dropout=0.15, bidirectional=False)
        >>>inp_tensor = torch.rand([10, 10, 10])
        >>>out_tensor = model(inp_tensor)
        >>>print(out_tensor.shape)
        torch.Size([10, 10])
    """

    def __init__(self, input_size=40, hidden_size=80, output_size=4, num_layers=8, dropout=0.15, bidirectional=False):
        super(MusicLSTM, self).__init__()

        self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)

        self.MLP = None
        if not bidirectional:
            self.MLP = nn.Linear(hidden_size, output_size)
        if bidirectional:
            self.MLP = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out = x
        out, _ = self.LSTM(out)
        # print(out.shape)
        out = torch.mean(out, 1)
        # print(out.shape)
        out = self.MLP(out)
        return out
