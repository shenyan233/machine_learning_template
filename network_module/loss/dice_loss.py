import torch


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, targets):
        class_mask = targets.data.new(targets.size(0), self.n_classes).fill_(0)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        return class_mask

    def forward(self, input, target, weight=None, softmax=True):
        if softmax:
            input = torch.softmax(input, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = torch.tensor([1] * self.n_classes).type_as(input)
        assert input.size() == target.size(), 'predict {} & target {} shape do not match'.format(input.size(), target.size())

        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(input * target, dim=0)
        y_sum = torch.sum(target, dim=0)
        z_sum = torch.sum(input, dim=0)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - (weight * loss).mean()
        return loss