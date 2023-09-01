import torch
import torch.nn as nn
import torch.nn.functional as F

from global_param import min_value
from network_module.compute_utils import one_hot_encoder


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, label_smooth=0, gamma=2, size_average=True, cof=1):
        super(FocalLoss, self).__init__()
        self.label_smooth = label_smooth
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.cof = cof

    def forward(self, inputs, targets):
        P = F.softmax(inputs, dim=1)

        class_mask = one_hot_encoder(targets, inputs.size(1))
        if self.label_smooth > 0:
            min_p = self.label_smooth / (self.class_num - 1)
            class_mask = (class_mask == 1) * (1 - self.label_smooth) + (class_mask == 0) * min_p

        batch_loss = (-class_mask * (torch.pow((1 - P), self.gamma)) * (P + min_value).log()).sum(1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
