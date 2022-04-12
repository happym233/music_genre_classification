import torch
import torch.nn as nn


class MLP(nn.Module):

    """
        MLP

        Arguments
        ---------
        input_dim: str
            input dimension
        output_dim: str
            output dimension
        hidden_dims: list
            an array of hidden layer dimensions
        activation: 'leakyReLU' if use leakyReLU as activation function
            else use ReLU as activation function
        batch_norm: boolean
            true if apply batch norm

        Example
        ---------
        >>> model = MLP(10, 10, [50])
        >>> inp_tensor = torch.rand([10, 10])
        >>> out_tensor = model(inp_tensor)
        >>> print(out_tensor.shape)

    """

    def __init__(self, input_dim, output_dim, hidden_dims=[], activation='relu', batch_norm=False):
        super(MLP, self).__init__()
        linear_array = []

        if activation.lower() == 'leakyrelu':
            activation_func = nn.LeakyReLU(0.01)
        elif activation.lower() == 'relu' or activation == '':
            activation_func = nn.ReLU()
        else:
            raise Exception('Unknown activation function')

        if hidden_dims is None or len(hidden_dims) == 0:
            linear_array.append(nn.Linear(input_dim, output_dim))
        else:
            hidden_input_dim = input_dim
            for hidden_output_dim in hidden_dims:
                linear_array.append(nn.Linear(hidden_input_dim, hidden_output_dim))
                cur = hidden_output_dim
                linear_array.append(activation_func)
                if batch_norm:
                    linear_array.append(nn.BatchNorm1d(cur))
            linear_array.append(nn.Linear(cur, output_dim))

        self.seq = nn.Sequential(*linear_array)

    def forward(self, x):
        return self.seq(x)
