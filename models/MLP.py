import torch
import torch.nn as nn


class MLP(nn.Module):
    """
        input_dim: input dimension
        output_dim: output dimension
        hidden_dims: an array of hidden layer dimensions
        activation: 'leakyReLU' if use leakyReLU as activation function
            else use ReLU as activation function
    """

    def __init__(self, input_dim, output_dim, hidden_dims=[], activation='relu'):
        super(MLP, self).__init__()
        linear_array = []

        if activation.lower() == 'leakyrelu':
            activation_func = nn.LeakyReLU(0.01)
        else:
            activation_func = nn.ReLU()

        if hidden_dims is None or len(hidden_dims) == 0:
            linear_array.append(nn.Linear(input_dim, output_dim))
        else:
            cur = input_dim
            for hidden_dim in hidden_dims:
                linear_array.append(nn.Linear(cur, hidden_dim))
                cur = hidden_dim
                linear_array.append(activation_func)
            linear_array.append(nn.Linear(cur, output_dim))
        self.seq = nn.Sequential(*linear_array)

    def forward(self, x):
        return self.seq(x)
