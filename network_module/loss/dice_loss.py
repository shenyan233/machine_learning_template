import torch

from global_param import min_value
from network_module.compute_utils import one_hot_encoder


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes, label_smooth=0, cof=1):
        super(DiceLoss, self).__init__()
        self.label_smooth = label_smooth
        self.n_classes = n_classes
        self.cof = cof

    def forward(self, input, target, weight=None, softmax=True):
        if softmax:
            input = torch.softmax(input, dim=1)
        target = one_hot_encoder(target, self.n_classes)
        if self.label_smooth > 0:
            min_p = self.label_smooth / (self.n_classes - 1)
            target = (target == 1) * (1 - self.label_smooth) + (target == 0) * min_p

        if weight is None:
            weight = torch.tensor([1] * self.n_classes).type_as(input)
        assert input.size() == target.size(), 'predict {} & target {} shape do not match'.format(input.size(),
                                                                                                 target.size())

        target = target.float()
        intersect = torch.sum(input * target, dim=0)
        y_sum = torch.sum(target, dim=0)
        z_sum = torch.sum(input, dim=0)
        loss = (2 * intersect + min_value) / (z_sum + y_sum + min_value)
        loss = 1 - (weight * loss).mean()
        return loss
