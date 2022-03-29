import torch

'''
    calculate the accuracy of a machine learning model
    model: the trained machine learning model
    X: input features
    y: the label of input features
    onthot: true if y is one-hot encoding, else y is ordinal encoding
'''


def cal_accuracy(model, X, y, onehot=False):
    pred = model(X)
    _, pred_ = torch.max(pred, 1)
    if onehot:
        _, res = torch.max(y, 1)
    else:
        res = y
    correct = (pred_ == res).sum()
    return correct / res.shape[0]
