import torch


def fast_hist(pred, label, n_classes):
    # confusion matrix
    # np.bincount counts the number of occurrences of each of n**2 numbers from 0 to n**2-1. return value shape (n, n)
    # The horizontal axis is the label and the vertical axis is the predicted value。
    # 混淆矩阵
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    # 横轴为标签，竖轴为预测值
    return torch.bincount(n_classes * label + pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def get_acc(hist):
    return torch.sum(torch.diag(hist)) / torch.sum(hist)


def get_precision(hist):
    return torch.diag(hist) / torch.sum(hist, dim=0)


def get_recall(hist):
    return torch.diag(hist) / torch.sum(hist, dim=1)


def get_f_score(hist, beta=1):
    precision = get_precision(hist)
    recall = get_recall(hist)
    return (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def get_acc_without_background(hist):
    # If the background is regarded as a positive case, the return value is the recall rate of the negative case
    # 得到除背景外的准确率，如果将背景视为正例，则返回值为负例的召回率
    return torch.sum(torch.diag(hist[1:, 1:])) / torch.sum(hist[1:, :])
