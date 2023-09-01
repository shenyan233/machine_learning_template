import pandas
import torch
from global_param import min_value
from network_module.compute_utils import one_hot_encoder


def fast_hist(pred, label, n_classes):
    # confusion matrix
    # np.bincount counts the number of occurrences of each of n**2 numbers from 0 to n**2-1. return value shape (n, n)
    # The vertical axis is the label and the horizontal axis is the predicted value。
    # 混淆矩阵
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    # 竖轴为标签，横轴为预测值
    return torch.bincount(n_classes * label + pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)


class Accuracy:
    name = 'acc'
    best_mode = 'max'

    def __init__(self, do_record=False):
        self.do_record = do_record

    def evaluate(self, output, target):
        """Computes the precision@k for the specified values of k"""
        output = output.softmax(1)
        topk = (1,)

        maxk = max(topk)
        batch_size = target.size(0)

        prob, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        if self.do_record:
            valid_prob = prob[((pred == 0) + (pred == 2)).t()]
            right_prob = prob[((pred == 0) + (pred == 2)).t() * correct.t()]
            false_prob = prob[((pred == 0) + (pred == 2)).t() * ~correct.t()]

            pandas.DataFrame(valid_prob.cpu().numpy()).to_csv('valid_prob.csv', header=False, index=False, mode='a')
            pandas.DataFrame(right_prob.cpu().numpy()).to_csv('right_prob.csv', header=False, index=False, mode='a')
            pandas.DataFrame(false_prob.cpu().numpy()).to_csv('false_prob.csv', header=False, index=False, mode='a')

        result = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            result.append(correct_k.mul_(100.0 / batch_size))
        return result[0]


class Precision:
    name = 'precision'
    best_mode = 'max'

    def __init__(self, class_weight=None):
        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight)

    def evaluate(self, output, target):
        output_index = output.argmax(1)
        output_one_hot = one_hot_encoder(output_index, output.size(1))
        target_one_hot = one_hot_encoder(target, output.size(1))

        tp = torch.sum(output_one_hot * target_one_hot, dim=0)  # 计算True Positive（预测为1且标签为1）的数量
        fp = torch.sum(output_one_hot * (1 - target_one_hot), dim=0)  # 计算False Positive（预测为1但标签为0）的数量

        precision = tp / (tp + fp + min_value)

        if self.class_weight is None:
            return precision.mean()
        else:
            return torch.sum(precision * self.class_weight) / torch.sum(self.class_weight)


class F1score:
    name = 'f1'
    best_mode = 'max'

    def evaluate(self, output, target):
        output_index = output.argmax(1)
        output_one_hot = one_hot_encoder(output_index, output.size(1))
        target_one_hot = one_hot_encoder(target, output.size(1))

        tp = torch.sum(output_one_hot * target_one_hot, dim=0)  # 计算True Positive（预测为1且标签为1）的数量
        fp = torch.sum(output_one_hot * (1 - target_one_hot), dim=0)  # 计算False Positive（预测为1但标签为0）的数量
        fn = torch.sum((1 - output_one_hot) * target_one_hot, dim=0)  # 计算False Negative（预测为0但标签为1）的数量

        f1 = (2 * tp) / (2 * tp + fn + fp + min_value)
        return f1.mean()
