import pytorch_lightning as pl
from torch import nn
import torch
from torchmetrics.classification.accuracy import Accuracy
from network.MLP import MLP


class TrainModule(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.time_sum = None
        self.config = config
        self.net = MLP(config['dim_in'], config['dim'], config['res_coef'], config['dropout_p'], config['n_layers'])
        # TODO 修改网络初始化方式为kaiming分布或者xavier分布
        self.loss = nn.BCELoss()
        self.accuracy = Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.net(x)
        loss = self.loss(x, y.type(torch.float32))
        acc = self.accuracy(x, y)
        self.log("Training loss", loss)
        self.log("Training acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.net(x)
        loss = self.loss(x, y.type(torch.float32))
        acc = self.accuracy(x, y)
        self.log("Validation loss", loss)
        self.log("Validation acc", acc)
        return loss, acc

    def test_step(self, batch, batch_idx):
        return 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def load_pretrain_parameters(self):
        """
        载入预训练参数
        """
        pass
