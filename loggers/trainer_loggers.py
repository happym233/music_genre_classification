import torch


def cal_accuracy(model, X, y, onehot=False):
    pred = model(X)
    _, pred_ = torch.max(pred, 1)
    if onehot:
        _, res = torch.max(y, 1)
    else:
        res = y
    correct = (pred_ == res).sum()
    return correct / res.shape[0]
