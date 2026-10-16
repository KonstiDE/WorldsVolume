import torch

def mae(pred, target):
    return torch.abs(pred - target).mean()


def medae(pred, target):
    return torch.abs(pred - target).median()


def r_mse(pred, target):
    mse = torch.square(pred - target).mean()
    return mse, torch.sqrt(mse)