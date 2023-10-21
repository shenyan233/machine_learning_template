import time
import pytorch_lightning as pl
import torch
from network.res_net.res_net import resnet56
from network_module.metric import Accuracy


class TrainModule(pl.LightningModule):
    time_sum = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = resnet56()
        self.losses = [torch.nn.CrossEntropyLoss()]
        self.evaluate = Evaluate()

    def forward(self, x):
        logits = self.net(x)
        return logits

    def on_train_start(self) -> None:
        lr_decay = self.config['lr_decay']
        self.trainer.optimizers[0].param_groups[0]['lr'] *= lr_decay
        now = self.trainer.optimizers[0].param_groups[0]['lr']
        if lr_decay != 1:
            print(f'lr multiplied by {lr_decay}, now is {now}')

    def step(self, stage: str, batch):
        _, x, label = batch
        if self.time_sum is None:
            time_start = time.time()
            logits = self.forward(x)
            time_end = time.time()
            self.time_sum = time_end - time_start
            self.log(f"{stage} time", self.time_sum)
        else:
            logits = self.forward(x)
        loss = sum([loss(logits, label) for loss in self.losses])
        assert not torch.isnan(loss)
        self.log(f"{stage} loss", loss)
        for metric in self.evaluate.evaluate:
            evaluation = metric.evaluate(logits, label)
            self.log(f"{stage} {metric.name}", evaluation)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step('Train', batch)

    def validation_step(self, batch, batch_idx):
        return self.step('Validation', batch)

    def test_step(self, batch, batch_idx):
        return self.step('Test', batch)

    def configure_optimizers(self):
        if 'amsgrad' not in self.config:
            amsgrad = True
        else:
            amsgrad = self.config['amsgrad']
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=1e-4, amsgrad=amsgrad)
        return optimizer

    def load_pretrain_parameters(self):
        """
        Load the pretraining parameters
        载入预训练参数
        """
        pass


class Evaluate:
    def __init__(self):
        # The first is the monitor
        self.evaluate = [Accuracy()]
