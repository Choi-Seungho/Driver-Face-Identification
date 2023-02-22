import torch
import numpy as np


def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()

def softmax(x):
    exp_a = np.exp(x)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp
    return y