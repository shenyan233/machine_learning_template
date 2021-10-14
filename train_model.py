import time

import numpy
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
import torch

from network.MLP_JDLU import MLP_JDLU
from network.MLP_ReLU import MLP_ReLU


class TrainModule(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.time_sum = None
        self.config = config
        if 1:
            self.net = MLP_ReLU(config['dim_in'], config['dim'], config['res_coef'], config['dropout_p'],
                                config['n_layers'])
        else:
            self.net = MLP_JDLU(config['dim_in'], config['dim'], config['res_coef'], config['dropout_p'],
                                config['n_layers'])
        self.loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.net(x)
        loss = self.loss(x, y.type(torch.float32))
        self.log("Training loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.net(x)
        loss = self.loss(x, y.type(torch.float32))
        self.log("Validation loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        input, label = batch
        if self.time_sum is None:
            time_start = time.time()
            pred = self.net(input)
            time_end = time.time()
            self.time_sum = time_end - time_start
            print(f'\n推理时间为: {self.time_sum:f}')
        else:
            pred = self.net(input)
        loss = self.loss(pred.reshape(1), label.type(torch.float32))
        self.log("Test loss", loss)
        return input, label, pred

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        records = numpy.empty((self.config['dataset_len'], 4))
        # count
        for cou in range(len(outputs)):
            records[cou, 0] = outputs[cou][0][0, 0]
            records[cou, 1] = outputs[cou][0][0, 1]
            records[cou, 2] = outputs[cou][1][0]
            records[cou, 3] = outputs[cou][2]

        import plotly.graph_objects as go
        trace0 = go.Mesh3d(x=records[:, 0],
                           y=records[:, 1],
                           z=records[:, 2],
                           opacity=0.5,
                           name='label'
                           )
        trace1 = go.Mesh3d(x=records[:, 0],
                           y=records[:, 1],
                           z=records[:, 3],
                           opacity=0.5,
                           name='pred'
                           )
        fig = go.Figure(data=[trace0, trace1])
        fig.update_layout(
            scene=dict(
                # xaxis=dict(nticks=4, range=[-100, 100], ),
                # yaxis=dict(nticks=4, range=[-50, 100], ),
                # zaxis=dict(nticks=4, range=[-100, 100], ),
                aspectratio=dict(x=1, y=1, z=0.5),
            ),

        )
        fig.show()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def load_pretrain_parameters(self):
        """
        载入预训练参数
        """
        pass
