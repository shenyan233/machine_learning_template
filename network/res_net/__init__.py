from network.res_net.res_net import resnet56
from torch import nn


def get_network():
    return resnet56()


def get_loss():
    return nn.CrossEntropyLoss()


def get_evaluation():
    return 'acc', 'max', evaluate


def evaluate(output, target):
    """Computes the precision@k for the specified values of k"""
    topk = (1,)

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))
    return result[0]
