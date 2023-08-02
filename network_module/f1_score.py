import torch
from network_module.compute_utils import one_hot_encoder


class Evaluate:
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