import pandas
import torch
from network_module.compute_utils import one_hot_encoder


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

        f1 = (2 * tp) / (2 * tp + fn + fp + 1e-8)
        return f1.mean()


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

            pandas.DataFrame(valid_prob.numpy()).to_csv('valid_prob.csv', header=False, index=False, mode='a')
            pandas.DataFrame(right_prob.numpy()).to_csv('right_prob.csv', header=False, index=False, mode='a')
            pandas.DataFrame(false_prob.numpy()).to_csv('false_prob.csv', header=False, index=False, mode='a')


class Precision:
    name = 'precision'
    best_mode = 'max'

    def __init__(self, class_weight=None):
        self.class_weight = torch.tensor(class_weight)

    def evaluate(self, output, target):
        output_index = output.argmax(1)
        output_one_hot = one_hot_encoder(output_index, output.size(1))
        target_one_hot = one_hot_encoder(target, output.size(1))

        tp = torch.sum(output_one_hot * target_one_hot, dim=0)  # 计算True Positive（预测为1且标签为1）的数量
        fp = torch.sum(output_one_hot * (1 - target_one_hot), dim=0)  # 计算False Positive（预测为1但标签为0）的数量

        precision = tp / (tp + fp + 1e-8)

        if precision.device != self.class_weight.device:
            self.class_weight = self.class_weight.type_as(precision)
        if self.class_weight is None:
            return precision.mean()
        else:
            return torch.sum(precision * self.class_weight) / torch.sum(self.class_weight)
