import time
import pytorch_lightning as pl
import torch
from torch import nn
from network.res_net.res_net import resnet56


class TrainModule(pl.LightningModule):
    time_sum = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = resnet56()
        self.loss = nn.CrossEntropyLoss()
        self.evaluate = Evaluate()

    def forward(self, input):
        pred = self.net(input)
        return pred

    def training_step(self, batch, batch_idx):
        _, input, label = batch
        pred = self.forward(input)
        loss = self.loss(pred, label)
        self.log("Training loss", loss)
        evaluation = self.evaluate.evaluate(pred, label)
        self.log(f"Training {self.evaluate.name}", evaluation)
        # TODO log learning rate
        return loss

    def validation_step(self, batch, batch_idx):
        _, input, label = batch
        pred = self.forward(input)
        loss = self.loss(pred, label)
        self.log("Validation loss", loss)
        # TODO modify the method of computing evaluation index
        evaluation = self.evaluate.evaluate(pred, label)
        self.log(f"Validation {self.evaluate.name}", evaluation)
        return loss

    def test_step(self, batch, batch_idx):
        _, input, label = batch
        if self.time_sum is None:
            time_start = time.time()
            pred = self.forward(input)
            time_end = time.time()
            self.time_sum = time_end - time_start
            print(f'\nInference time is {self.time_sum:f}|推理时间为: {self.time_sum:f}')
        else:
            pred = self.forward(input)
        loss = self.loss(pred, label)
        self.log("Test loss", loss)
        evaluation = self.evaluate.evaluate(pred, label)
        self.log(f"Test {self.evaluate.name}", evaluation)
        return loss

    def configure_optimizers(self):
        lr = 0.1
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)
        # TODO 实现warmup
        return [optimizer], [lr_scheduler]

    def load_pretrain_parameters(self):
        """
        Load the pretraining parameters
        载入预训练参数
        """
        pass


class Evaluate:
    name = 'acc'
    best_mode = 'max'

    def evaluate(self, output, target):
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

