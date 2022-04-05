import torch

'''
    calculate the accuracy of a machine learning model
    model: the trained machine learning model
    X: input features
    y: the label of input features
    onthot: true if y is one-hot encoding, else y is ordinal encoding
'''


def cal_accuracy(model, X, y, with_batch=False, batch_size=100):
    if not with_batch:
        pred = model(X)
        _, pred_ = torch.max(pred, 1)
        correct = (pred_ == y).sum()
        return correct / y.shape[0]
    else:
        correct = 0
        for i in range(0, len(X), batch_size):
            _input = X[i:i + batch_size]
            _label = y[i:i+batch_size]
            pred = model(_input)
            _, pred_ = torch.max(pred, 1)
            correct = correct + (pred_ == _label).sum()
        return correct / y.shape[0]
