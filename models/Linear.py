import torch


class LogisticRegression(torch.nn.Module):
    """
        Logistic regression

        Arguments
        ---------
        input_dim: input dimension
        output_dim: output dimension

        Example
        ---------
        >>> model = (10, 10)
        >>> inp_tensor = torch.rand([10, 10])
        >>> out_tensor = model(inp_tensor)
        >>> print(out_tensor.shape)
        torch.Size([10, 10])
    """
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
