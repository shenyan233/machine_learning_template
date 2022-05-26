import importlib
import time

import numpy
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
import torch


class TrainModule(pl.LightningModule):
    time_sum = None
    quantile25 = None
    quantile75 = None
    quantile50 = None
    acc_min = None
    acc_max = None

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        imported = importlib.import_module('network.%(model_name)s' % config)
        # TODO Modify the network and loss function
        self.net = imported.resnet56()
        self.accuracy = imported.accuracy
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        pred = self.net(input)
        return pred

    def training_step(self, batch, batch_idx):
        _, input, label = batch
        label = label.flatten()
        pred = self.net(input)
        loss = self.loss(pred, label)
        self.log("Training loss", loss)
        acc = self.accuracy(pred, label)[0]
        self.log("Training acc", acc)
        # TODO log learning rate
        return loss

    def validation_step(self, batch, batch_idx):
        _, input, label = batch
        label = label.flatten()
        pred = self.net(input)
        loss = self.loss(pred, label)
        self.log("Validation loss", loss)
        acc = self.accuracy(pred, label)[0]
        self.log("Validation acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        _, input, label = batch
        label = label.flatten()
        if self.time_sum is None:
            time_start = time.time()
            pred = self.net(input)
            time_end = time.time()
            self.time_sum = time_end - time_start
            print(f'\nInference time is {self.time_sum:f}|推理时间为: {self.time_sum:f}')
        else:
            pred = self.net(input)
        loss = self.loss(pred, label)
        self.log("Test loss", loss)
        acc = self.accuracy(pred, label)[0]
        self.log("Test acc", acc)
        return input, label, pred

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        accs = numpy.zeros((1, len(outputs)))
        for i in range(len(outputs)):
            accs[0, i] = outputs[i][2].cpu().numpy()
        self.quantile25 = numpy.quantile(accs, 0.25)
        self.quantile50 = numpy.quantile(accs, 0.5)
        self.quantile75 = numpy.quantile(accs, 0.75)
        self.acc_min = numpy.min(accs)
        self.acc_max = numpy.max(accs)

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
