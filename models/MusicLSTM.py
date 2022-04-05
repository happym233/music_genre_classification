import torch
import torch.nn as nn
from .MLP import MLP


class MusicLSTM(nn.Module):
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
