import torch


def cal_accuracy(model, X, y):
    pred = model(X)
    _, pred_ = torch.max(pred, 1)
    _, res = torch.max(y, 1)
    correct = (pred_ == res).sum()
    return correct / res.shape[0]
