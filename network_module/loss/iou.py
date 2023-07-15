"""
Functions and classes related to calculating iOU, including calculating IOU Loss
和计算iou相关的函数和类, 包括计算iou loss
"""
import torch
from torch import nn

from network_module.classify_eval import fast_hist
from network_module.compute_utils import torch_nanmean


def per_class_iou(hist):
    # A one-dimensional array of the diagonal values of the HIST matrix/Sum of all elements of hIST matrix
    # return value shape (n,)
    # List.sum (0)= Add in columns List.sum (1) Is added in rows,
    # where the rows represent labels and the columns represent predictions
    # hist矩阵的对角线上的值组成的一维数组/hist矩阵的所有元素之和，返回值形状(n,)
    # hist.sum(0)=按列相加  hist.sum(1)按行相加, 行表示标签, 列表示预测
    return (torch.diag(hist)) / (torch.sum(hist, 1) + torch.sum(hist, 0) - torch.diag(hist))


def get_ious(pred, label, n_classes, softmax=True):
    if softmax:
        pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    hist = fast_hist(pred.flatten(), label.flatten(), n_classes)
    IoUs = per_class_iou(hist)
    mIoU = torch_nanmean(IoUs[1:n_classes])
    return mIoU, IoUs


class IOULoss(nn.Module):
    def __init__(self, n_classes):
        super(IOULoss, self).__init__()
        self.n_classes = n_classes

    def forward(self, pred, label):
        mIoU, _ = get_ious(pred, label, self.n_classes)
        return 1 - mIoU


class IOU:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.hist = None

    def add_data(self, preds, label):
        self.hist = torch.zeros((self.n_classes, self.n_classes)).type_as(
            preds) if self.hist is None else self.hist + fast_hist(preds.int(), label, self.n_classes)

    def get_miou(self):
        IoUs = per_class_iou(self.hist)
        self.hist = None
        mIoU = torch_nanmean(IoUs[1:self.n_classes])
        return mIoU, IoUs
