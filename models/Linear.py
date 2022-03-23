import torch


class LogisticRegression(torch.nn.Module):
    '''
        input_dim: input dimension
        output_dim: output dimension
    '''
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
