import importlib
import time
import pytorch_lightning as pl
import torch


class TrainModule(pl.LightningModule):
    time_sum = None

    def __init__(self, config=None):
        super().__init__()
        self.config = config
        imported = importlib.import_module('network.%(model_name)s' % config)
        self.net = imported.get_network()
        self.loss = imported.get_loss()
        self.monitor, _, self.evaluate = imported.get_evaluation()

    def forward(self, input):
        pred = self.net(input)
        return pred

    def training_step(self, batch, batch_idx):
        _, input, label = batch
        pred = self.forward(input)
        loss = self.loss(pred, label)
        self.log("Training loss", loss)
        evaluation = self.evaluate(pred, label)
        self.log(f"Training {self.monitor}", evaluation)
        # TODO log learning rate
        return loss

    def validation_step(self, batch, batch_idx):
        _, input, label = batch
        pred = self.forward(input)
        loss = self.loss(pred, label)
        self.log("Validation loss", loss)
        evaluation = self.evaluate(pred, label)
        self.log(f"Validation {self.monitor}", evaluation)
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
        evaluation = self.evaluate(pred, label)
        self.log(f"Test {self.monitor}", evaluation)
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
